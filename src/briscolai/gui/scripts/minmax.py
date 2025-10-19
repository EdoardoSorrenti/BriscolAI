import random
import torch
from pathlib import Path
from copy import copy

LOOKUP_PATH = Path(__file__).resolve().parent / "lookup_tables.pt"

VALUES_PER_RANK = torch.tensor([11, 0, 10, 0, 0, 0, 0, 2, 3, 4], dtype=torch.int32)
VALUES_PER_CARD = VALUES_PER_RANK.repeat(4)


_lookup_payload = torch.load(LOOKUP_PATH)
WINNER_TABLE = _lookup_payload["winner"].to(torch.int8)


def evaluate(card1, card2, briscola, player2_starts):
    relative_winner = WINNER_TABLE[briscola, player2_starts, card1, card2].item()
    if relative_winner == 0:
        return player2_starts
    return 1 - player2_starts

def get_points(card1, card2):
    return VALUES_PER_CARD[card1].item() + VALUES_PER_CARD[card2].item()


def minmax(hands, briscola, player2_starts, ontable=None, points = 0, first_call = True): # hands structure: [[p1_cards], [p2_cards]], briscola 0-3, player2_starts 0/1
    if len(hands[0]) + len(hands[1]) == 0:
        return points
    options = []
    candidate_cards = []
    if ontable is None:
        for card in hands[player2_starts]:
            candidate_cards.append(card)
            new_hands = [hand.copy() for hand in hands]
            new_hands[player2_starts].remove(card)
            options.append(minmax(new_hands, briscola, player2_starts, ontable=card, points=points, first_call=False))
    else:
        for card in hands[1-player2_starts]:
            candidate_cards.append(card)
            new_hands = [hand.copy() for hand in hands]
            new_hands[1-player2_starts].remove(card)
            winner = evaluate(ontable, card, briscola, player2_starts)
            new_turn = winner
            new_points = points + get_points(ontable, card) * (1 if winner == 0 else -1)
            options.append(minmax(new_hands, briscola, new_turn, ontable=None, points=new_points, first_call=False))
    if (player2_starts == 0 and ontable is None) or (player2_starts == 1 and ontable is not None):
        best_option = max(options)
    else:
        best_option = min(options)
    
    if first_call:
        best_index = options.index(best_option)
        return candidate_cards[best_index]
    else:
        return best_option
    
if __name__ == "__main__":
    deck = list(range(40))
    random.shuffle(deck)

    hand1 = [deck.pop() for _ in range(3)]
    hand2 = [deck.pop() for _ in range(3)]
    briscola = random.randint(0,3)
    player2_starts = random.randint(0,1)

    print("Hand 1:", hand1)
    print("Hand 2:", hand2)
    print("Briscola:", briscola)
    print("Player 2 starts:", player2_starts)
    card1 = minmax([hand1, hand2], briscola, player2_starts)
    print("Minmax chooses:", card1)
