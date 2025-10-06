import torch

def card_to_onehot(card):
    """Converts a card ID to a onehot tensor."""
    onehot = torch.zeros(40, dtype=torch.bool)
    if card is not None:
        onehot[card] = 1 # Card IDs are 0-39
    return onehot

def cards_to_multihot(cards):
    """Converts a list of card IDs to a multihot tensor."""
    multihot = torch.zeros(40, dtype=torch.bool)
    for card in cards:
        multihot[card] = 1 # Card IDs are 0-39
    return multihot

def get_state(session, player_id=0, on_table=None):
    """Returns the current state of the game as a tensor."""
    opponent_id = 1 - player_id
    
    my_hand = cards_to_multihot(session.hands[player_id])
    my_taken = cards_to_multihot(session.taken[player_id])
    opponent_taken = cards_to_multihot(session.taken[opponent_id])
    briscola = card_to_onehot(session.briscola)
    on_table = card_to_onehot(on_table) 

    return torch.cat([my_hand, my_taken, opponent_taken, briscola, on_table], dim=0).float().unsqueeze(0)  # Shape: (1, 200)

def get_action_mask(session, player_id=0):
    """Returns a mask of valid actions for the current player."""
    hand = session.hands[player_id]
    mask = torch.zeros(40, dtype=torch.bool)
    for card in hand:
        mask[card] = 1
    return mask.unsqueeze(0)  # Shape: (1, 40)