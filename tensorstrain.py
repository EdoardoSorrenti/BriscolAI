from tensorgames import Games
from time import perf_counter
import torch
from model import PolicyNetwork
from config import *

device = torch.device(device_name)

torch.set_default_device(device)

model= PolicyNetwork()
model.to(device)

try:
    model.load_state_dict(torch.load(save_path))
    print(f"Loaded model from {save_path}")
except FileNotFoundError:
    print(f"No saved model found at {save_path}, starting fresh.")

try:
    model = torch.compile(model, fullgraph=False, backend="eager")
except Exception as e:
    print(f"Warning: torch.compile failed with error {e}, continuing without it.")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def random_pick_multihot(x):
    # Random numbers, set zeros where x is zero
    rand = torch.rand_like(x, dtype=torch.float) * x
    # Pick the argmax of random numbers (guaranteed to pick one of the nonzero entries)
    return rand.argmax(dim=1)

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

    logits = model(masks, states)
    action_dists = torch.distributions.Categorical(logits=logits)
    actions = action_dists.sample()
    log_probs = action_dists.log_prob(actions)
    return actions, log_probs


def tensorloop_model_vs_random(games, model):
    """
    Plays a batch of games with the model (P0) vs a random player (P1).
    Handles games where the model plays first and games where the random player plays first.
    """
    n_games = games.N
    game_indices = torch.arange(n_games, dtype=torch.int64)
    
    # We will store the log_probs for each game's trajectory
    all_log_probs = torch.zeros((n_games, 40), dtype=torch.float32)

    # Cards played in the current trick, initialized to an invalid value
    card1 = torch.full((n_games,), -1, dtype=torch.int64)
    card2 = torch.full((n_games,), -1, dtype=torch.int64)

    counter = 0

    for _ in range(20):  # 20 tricks per game
        # --- Determine whose turn it is for each game ---
        turn0_mask = games.turns == 0
        turn1_mask = games.turns == 1

        # --- Process games where Model (P0) plays first ---
        if turn0_mask.any():
            # Get moves from the model for this subset of games
            actions, log_probs = get_moves(games, model, player_id=0, mask=turn0_mask)
            card1[turn0_mask] = actions
            
            all_log_probs[game_indices[turn0_mask], counter] = log_probs

            # Random player (P1) responds
            card2[turn0_mask] = random_pick_multihot(games.hands[1][turn0_mask])

        # --- Process games where Random Player (P1) plays first ---
        if turn1_mask.any():
            # Random player (P1) plays
            card2[turn1_mask] = random_pick_multihot(games.hands[1][turn1_mask])
            
            # Update the 'on_table' state so the model sees the opponent's card
            games.on_table[game_indices[turn1_mask], card2[turn1_mask]] = 1

            # Get moves from the model for this subset of games
            actions, log_probs = get_moves(games, model, player_id=0, mask=turn1_mask)
            card1[turn1_mask] = actions

            all_log_probs[game_indices[turn1_mask], counter] = log_probs

        # --- Update game state for all games at once ---
        games.hands[0].scatter_(1, card1.unsqueeze(1), 0)
        games.hands[1].scatter_(1, card2.unsqueeze(1), 0)

        # Compare cards and determine winners for the trick
        winners = games.compare_cards(card1, card2)

        # Update taken cards based on who won
        p0_wins_mask = winners == 0
        p1_wins_mask = winners == 1

        if p0_wins_mask.any():
            games.taken[0][game_indices[p0_wins_mask], card1[p0_wins_mask]] = 1
            games.taken[0][game_indices[p0_wins_mask], card2[p0_wins_mask]] = 1
        
        if p1_wins_mask.any():
            games.taken[1][game_indices[p1_wins_mask], card1[p1_wins_mask]] = 1
            games.taken[1][game_indices[p1_wins_mask], card2[p1_wins_mask]] = 1

        # Clear the table for the next trick
        games.on_table.zero_()
        
        # The winner of the trick leads the next one
        games.turns = winners.to(torch.int8)
        games.draw()
        counter += 1

    # --- Game finished, calculate results ---
    p1_scores, p2_scores, winners = games.check_winners()
    rewards = (p1_scores - p2_scores) / 60.0  # Normalize rewards

    log_prob_sums = all_log_probs.sum(dim=1)
    policy_losses = - log_prob_sums * rewards

    return policy_losses.sum(), winners


def train_model(batches, batch_size):
    """Trains the model for a given number of epochs."""
    start_time_total = perf_counter()
    wins = 0
    tot_games = 0
    
    games = Games(batch_size, alternate_turns=True)  # Wins, Losses, Draws
    
    for batch in range(batches):
        try:
            optimizer.zero_grad()
            games.reset()
            
            policy_loss, winners_array = tensorloop_model_vs_random(games, model)

            wins += (winners_array == 0)
            tot_games += batch_size


            if torch.is_tensor(policy_loss):
                policy_loss.backward()
                optimizer.step()

            if batch % log_freq == 0 and batch > 0:
                p1s, p2s = games.count_points()
                
                end_time = perf_counter()
                tot_time = end_time - start_time_total
                games_processed = batch_size * log_freq
                games_per_second = games_processed / tot_time if tot_time > 0 else 0
                
                avg_p1 = p1s.mean().item()
                avg_p2 = p2s.mean().item()
                win_rate = (wins.sum().item() / tot_games) * 100

                print(f"\n--- Batch {batch} ---")
                print(f"Model Win Rate: {win_rate:.1f}%")
                print(f"Avg Points (Model vs Random): {avg_p1:.1f} vs {avg_p2:.1f}")
                print(f"Compute Speed: {round(games_per_second)} games/sec")
                if torch.is_tensor(policy_loss):
                    print(f"Policy Loss: {policy_loss.item():.4f}")
                
                start_time_total = perf_counter()

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