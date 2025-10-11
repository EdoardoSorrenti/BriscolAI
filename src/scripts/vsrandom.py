from time import perf_counter
import torch
from trainconfig import *


torch.set_default_device(device)
torch.set_default_dtype(dtype)


from model import PolicyNetwork
from tensorgames import Games


torch.set_default_device(device)

model= PolicyNetwork()

try:
    model.load_state_dict(torch.load(save_path))
    print(f"Loaded model from {save_path}")
except FileNotFoundError:
    print(f"No saved model found at {save_path}, starting fresh.")

try:
    model = torch.compile(model, fullgraph=False)
except Exception as e:
    print(f"Warning: torch.compile failed with error {e}, continuing without it.")


model.to(device, dtype=dtype)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def get_states(session, player_id=0):
    """Returns the current state of the game as a tensor."""
    opponent_id = 1 - player_id
    
    my_hand = session.hands[player_id]
    my_taken = session.taken[player_id]
    opponent_taken = session.taken[opponent_id]
    briscole = session.briscole_cards
    on_table = session.on_table

    return torch.cat([my_hand, my_taken, opponent_taken, briscole, on_table], dim=1)  # Shape: (N, 200)

def get_action_masks(session, player_id=0):
    """Returns a mask of valid actions for the current player."""
    return session.hands[player_id].bool()  # Shape: (N, 40)

def get_moves(session, model, player_id=0, mask=None):
    """Returns a move for the current player using the model."""
    states = get_states(session, player_id)
    masks = get_action_masks(session, player_id)

    if mask is not None:
        states = states[mask]
        masks = masks[mask]

    probs = model(states, masks)

    # Work in float32 for stable sampling/logarithms, then return the log-probs
    # in the default dtype. Masking is already applied inside the model.
    probs_fp32 = probs.to(torch.float32)
    actions = torch.multinomial(probs_fp32, 1).squeeze(1)
    selected_probs = probs_fp32.gather(1, actions.unsqueeze(1)).squeeze(1)
    log_probs = selected_probs.clamp_min(1e-12).log()
    return actions, log_probs.to(probs.dtype)


def tensorloop_model_vs_random(games, model):
    """
    Plays a batch of games with the model (P0) vs a random player (P1).
    Handles games where the model plays first and games where the random player plays first.
    """
    n_games = games.N
    # We will store the log_probs for each game's trajectory (20 moves per game)
    all_log_probs = torch.zeros((n_games, 20))

    # Cards played in the current trick, initialized to an invalid value
    card1 = torch.full((n_games,), -1, dtype=torch.int64)
    card2 = torch.full((n_games,), -1, dtype=torch.int64)

    for i in range(20):  # 20 tricks per game
        # --- Determine whose turn it is for each game ---
        (turn0_mask, turn0_idx), (turn1_mask, turn1_idx) = games.split_turns()
        games.refresh_random_buffer()

        # --- Process games where Model (P0) plays first ---
        if turn0_idx.numel():
            actions0, log_probs0 = get_moves(games, model, player_id=0, mask=turn0_mask)
            card1.index_copy_(0, turn0_idx, actions0)

            games.place_on_table(turn0_idx, actions0)  # expose card to opponent
            all_log_probs[turn0_idx, i] = log_probs0

            card2_turn0 = games.pick_random_cards(player_id=1, indices=turn0_idx)
            card2.index_copy_(0, turn0_idx, card2_turn0)

        # --- Process games where Random Player (P1) plays first ---
        if turn1_idx.numel():
            card2_turn1 = games.pick_random_cards(player_id=1, indices=turn1_idx)
            card2.index_copy_(0, turn1_idx, card2_turn1)

            games.place_on_table(turn1_idx, card2_turn1)  # model observes opponent card

            actions1, log_probs1 = get_moves(games, model, player_id=0, mask=turn1_mask)
            card1.index_copy_(0, turn1_idx, actions1)

            all_log_probs[turn1_idx, i] = log_probs1

        # --- Update game state for all games at once ---
        games.remove_played_cards(card1, card2)

        # Compare cards and determine winners for the trick
        winners = games.compare_cards(card1, card2)

        # Update taken cards based on who won
        games.record_taken_cards(winners, card1, card2)

        # Clear the table for the next trick
        games.clear_table()
        
        # The winner of the trick leads the next one
        games.draw()



    # --- Game finished, calculate results ---
    if optimize_points:
        p1s, p2s, winners = games.check_winners()
        rewards = (p1s - p2s) / 60  # Reward is the point difference
    else:
        _, _, winners = games.check_winners()
        rewards = torch.zeros_like(winners).masked_fill_(winners == 0, 1.0).masked_fill_(winners == 1, -1.0)

    log_prob_sums = all_log_probs.sum(dim=1)
    policy_losses = - log_prob_sums * rewards

    return policy_losses.mean(), winners


def train_model(batches, batch_size):
    """Trains the model for a given number of epochs."""
    start_time_total = perf_counter()
    wins = 0
    losses = 0
    tot_games = 0
    batch = 0
    
    games = Games(batch_size, alternate_turns=True)  # Wins, Losses, Draws
    
    while True:
        batch += 1
        try:
            optimizer.zero_grad()
            games.reset()
            
            policy_loss, winners_array = tensorloop_model_vs_random(games, model)


            if torch.is_tensor(policy_loss):
                policy_loss.backward()
                optimizer.step()

            if batch % log_freq == 0 and batch > 0:
                p1s, p2s, winners_array = games.check_winners()
                wins = winners_array.eq(0).sum()
                losses = winners_array.eq(1).sum()
                tot_games = winners_array.size(0)

                win_rate = (wins.sum().item() / tot_games) * 100
                loss_rate = (losses.sum().item() / tot_games) * 100
                
                end_time = perf_counter()
                tot_time = end_time - start_time_total
                games_processed = batch_size * log_freq
                games_per_second = games_processed / tot_time if tot_time > 0 else 0
                
                avg_p1 = p1s.mean().item()
                avg_p2 = p2s.mean().item()
                win_rate = (wins.sum().item() / tot_games) * 100
                loss_rate = (losses.sum().item() / tot_games) * 100

                print(f"\n--- Batch {batch} ---")
                print(f"Model Win Rate: {win_rate:.3f}%, Loss Rate: {loss_rate:.3f}%")
                print(f"Avg Points (Model vs Random): {avg_p1:.3f} vs {avg_p2:.3f}")
                print(f"Compute Speed: {round(games_per_second)} games/sec")

                if torch.is_tensor(policy_loss):
                    print(f"Policy Loss: {policy_loss.item()*1000:.4f}")
                
                start_time_total = perf_counter()
                wins = 0
                losses = 0
                tot_games = 0

            if batch % save_freq == 0 and batch > 0:
                if save_results:
                    if hasattr(model, '_orig_mod'):
                        torch.save(model._orig_mod.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving final model...")
            break

    # Save the final model
    if save_results:
        if hasattr(model, '_orig_mod'):
            torch.save(model._orig_mod.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")

if __name__ == '__main__':
    train_model(batches, batch_size)