from tensorgame import Game
from time import perf_counter
import torch
from model import PolicyNetwork
from random import choice

batches = 200
batch_size = 3300
learning_rate = 3e-3
gamma = 1.0  # No discounting

version = "0.2"

save_path = f'models/model_v{version}.pth'

log_freq = 1

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

def get_state(session, player_id=0):
    """Returns the current state of the game as a tensor."""
    opponent_id = 1 - player_id
    
    my_hand = session.hands[player_id]
    my_taken = session.taken[player_id]
    opponent_taken = session.taken[opponent_id]
    briscola = session.briscola_card
    on_table = session.on_table

    return torch.cat([my_hand, my_taken, opponent_taken, briscola, on_table], dim=0).unsqueeze(0)  # Shape: (1, 200)

def get_action_mask(session, player_id=0):
    """Returns a mask of valid actions for the current player."""
    return session.hands[player_id].bool().unsqueeze(0)  # Shape: (1, 40)

def get_move(session, model, player_id=0):
    """Returns a move for the current player using the model."""
    state = get_state(session, player_id)
    mask = get_action_mask(session, player_id)
    logits = model(mask, state)
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    card = action.item()
    return card, log_prob

def fastloop(session, model):
    """Plays a whole game with the model as player 0 and a random player as player 1."""
    log_probs = []
    while not session.check_finished():
        if session.turno == 0:  # Model's turn to play first
            card1, log_prob = get_move(session, model)
            session.hands[0][card1] = 0
            log_probs.append(log_prob)
            session.on_table[card1] = 1

            card2 = choice(session.hands[1].nonzero(as_tuple=True)[0].tolist())
            session.hands[1][card2] = 0
        else:
            card2 = choice(session.hands[1].nonzero(as_tuple=True)[0].tolist())
            session.hands[1][card2] = 0
            session.on_table[card2] = 1

            card1, log_prob = get_move(session, model)
            session.hands[0][card1] = 0
            log_probs.append(log_prob)

        winner = session.compare_hands(card1, card2)
        session.taken[winner][card1] = 1
        session.taken[winner][card2] = 1
        session.on_table.zero_()
        session.turno = winner
        session.draw()

    return session.check_winner(), torch.stack(log_probs)

def train_model(batches, batch_size):
    """Trains the model for a given number of epochs."""
    game = Game()
    start_time = perf_counter()
    try:
        for batch in range(batches):
            outcomes = [0, 0, 0]  # Wins for player 1, player 2, draws
            total_points = [0, 0]
            optimizer.zero_grad()
            policy_losses = []
            for episode in range(batch_size):
                game.reset(turno=episode % 2)
                winner, log_probs = fastloop(game, model)
                p1, p2 = game.count_points()
                outcomes[winner] += 1
                total_points[0] += p1
                total_points[1] += p2
                
                reward = (p1 - p2) / 60 # Normalize reward to be between -2 and 2
                policy_losses.append(-log_probs.sum() * reward)

            policy_loss = torch.stack(policy_losses).sum()
            policy_loss.backward()
            optimizer.step()

            if batch % log_freq == 0:
                end_time = perf_counter()
                tot_time = end_time - start_time
                games_per_second = (batch_size * log_freq) / tot_time
                avg_p1 = total_points[0]/batch_size
                avg_p2 = total_points[1]/batch_size
                print(f"Batch {batch}:")
                print(f"Winner Model: {outcomes[0]/batch_size*100:.1f}%, Winner Noob: {outcomes[1]/batch_size*100:.1f}%, draws: {outcomes[2]/batch_size*100:.1f}%")
                print(f"Avg points: Model {avg_p1:.1f}, Noob {avg_p2:.1f}")
                print(f"Compute speed: {round(games_per_second)} games/second")
                start_time = perf_counter()
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")

    # Save the model
    if hasattr(model, '_orig_mod'):
        torch.save(model._orig_mod.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train_model(batches, batch_size)