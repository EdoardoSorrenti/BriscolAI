from tensorgames import Games
from time import perf_counter
import torch
from model import PolicyNetwork
from random import choice
from config import *
import random

model= PolicyNetwork()

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

def card_to_onehot(card):
    """Converts a card ID to a onehot tensor."""
    onehot = torch.zeros(40, dtype=torch.float32)
    if card is not None:
        onehot[card] = 1 # Card IDs are 0-39
    return onehot

def cards_to_multihot(cards):
    """Converts a list of card IDs to a multihot tensor."""
    multihot = torch.zeros(40, dtype=torch.float32)
    for card in cards:
        multihot[card] = 1 # Card IDs are 0-39
    return multihot

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
    all_log_probs = [[] for _ in range(n_games)]

    # Cards played in the current trick, initialized to an invalid value
    card1 = torch.full((n_games,), -1, dtype=torch.int64)
    card2 = torch.full((n_games,), -1, dtype=torch.int64)

    while not games.check_finished():
        # --- Determine whose turn it is for each game ---
        turn0_mask = games.turns == 0
        turn1_mask = games.turns == 1

        # --- Process games where Model (P0) plays first ---
        if turn0_mask.any():
            # Get moves from the model for this subset of games
            actions, log_probs = get_moves(games, model, player_id=0, mask=turn0_mask)
            card1[turn0_mask] = actions
            
            # Store log_probs for the corresponding games
            for i, lp in zip(game_indices[turn0_mask], log_probs):
                all_log_probs[i].append(lp)

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

            # Store log_probs
            for i, lp in zip(game_indices[turn1_mask], log_probs):
                all_log_probs[i].append(lp)

        # --- Update game state for all games at once ---
        games.hands[0][game_indices, card1] = 0
        games.hands[1][game_indices, card2] = 0
        
        # The on_table for turn0_mask games wasn't updated yet
        games.on_table[game_indices, card1] = 1
        games.on_table[game_indices, card2] = 1

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

    # --- Game finished, calculate results ---
    p1_scores, p2_scores = games.count_points()
    rewards = (p1_scores - p2_scores) / 60.0  # Normalize rewards

    policy_losses = []
    for i in range(n_games):
        if all_log_probs[i]:  # If the model made any moves
            log_prob_sum = torch.stack(all_log_probs[i]).sum()
            policy_losses.append(-log_prob_sum * rewards[i])

    return torch.stack(policy_losses).sum(), games.check_winners()


def train_model(batches, batch_size):
    """Trains the model for a given number of epochs."""
    start_time_total = perf_counter()
    
    for batch in range(batches):
        try:
            optimizer.zero_grad()
            
            # Alternate starting player for each batch
            games = Games(batch_size, alternate_turns=True)
            games.reset()
            
            policy_loss, winners_array = tensorloop_model_vs_random(games, model)
            
            if torch.is_tensor(policy_loss):
                policy_loss.backward()
                optimizer.step()

            if batch % log_freq == 0 and batch > 0:
                p1s, p2s = games.count_points()
                outcomes = [
                    (winners_array == 0).sum().item(),
                    (winners_array == 1).sum().item(),
                    (winners_array == 2).sum().item()
                ]
                
                end_time = perf_counter()
                tot_time = end_time - start_time_total
                games_processed = batch_size * log_freq
                games_per_second = games_processed / tot_time if tot_time > 0 else 0
                
                avg_p1 = p1s.mean().item()
                avg_p2 = p2s.mean().item()
                win_rate = (outcomes[0] / batch_size) * 100
                draw_rate = (outcomes[2] / batch_size) * 100

                print(f"\n--- Batch {batch} ---")
                print(f"Model Win/Draw %: {win_rate:.1f}% / {draw_rate:.1f}%")
                print(f"Avg Points (Model vs Random): {avg_p1:.1f} vs {avg_p2:.1f}")
                print(f"Compute Speed: {round(games_per_second)} games/sec")
                if torch.is_tensor(policy_loss):
                    print(f"Policy Loss: {policy_loss.item():.4f}")
                start_time_total = perf_counter()

            if batch % save_freq == 0 and batch > 0:
                if hasattr(model, '_orig_mod'):
                    torch.save(model._orig_mod.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving final model...")
            break

    # Save the final model
    if hasattr(model, '_orig_mod'):
        torch.save(model._orig_mod.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")

if __name__ == '__main__':
    train_model(batches, batch_size)