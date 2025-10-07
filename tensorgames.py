import torch
from config import device_name

device = torch.device(device_name)
torch.set_default_device(device)

# Point values assigned to each card
VALUES = {0:11, 1:0, 2:10, 3:0, 4:0, 5:0, 6:0, 7:2, 8:3, 9:4}  # Valori carte a briscola
VALUES_TENSOR = torch.tensor([11,0,10,0,0,0,0,2,3,4]*4, dtype=torch.float32, device=device)

# Creazione mazzo base
DECK = list(range(40))  # 0..39

class Games:
    def __init__(self, N, alternate_turns=True):
        """Initializes tensors for an empty game."""
        self.N = N
        self.alternate_turns = alternate_turns

        # Vectorized: (2, N, 40) so hands[0] -> player 1, hands[1] -> player 2
        self.hands  = torch.zeros((2, N, 40), dtype=torch.float32, device=device)  # Carte in mano
        self.taken  = torch.zeros((2, N, 40), dtype=torch.float32, device=device)  # Carte prese
        self.briscole_cards = torch.zeros((N, 40), dtype=torch.float32, device=device)
        self.on_table = torch.zeros((N, 40), dtype=torch.float32, device=device)

        # Utils
        self.turns = torch.zeros(N, dtype=torch.int8, device=device)     # 0 -> P0 first, 1 -> P1 first
        self.briscole = torch.zeros(N, dtype=torch.int8, device=device)  # suit (0..3)

        # Decks (initialized in reset)
        self.decks = None                 # (N, 40) int64
        self.deck_pos = None              # shared python int

    def reset(self, turno=0):
        """Resets the games to initial state."""
        self.hands.zero_()
        self.taken.zero_()
        self.briscole_cards.zero_()
        self.on_table.zero_()

        self.briscole.zero_()
        self.turns.zero_()

        # Create tensorized decks: each row a shuffled permutation of 0..39.
        self.decks = torch.stack([torch.randperm(40, device=device) for _ in range(self.N)])  # (N, 40), int64
        self.deck_pos = 39  # last index (shared across all rows)

        # Briscola suit determined by the "bottom" card (index 0)
        self.briscole = (self.decks[:, 0] // 10).to(dtype=torch.int8)

        # One-hot suits in briscole_cards via scatter_ (indices must be long)
        suit_idx = self.briscole.to(torch.long).unsqueeze(1)             # (N,1)
        self.briscole_cards.zero_()
        self.briscole_cards.scatter_(1, suit_idx, torch.ones((self.N,1), device=device))

        # Initial turns
        if self.alternate_turns:
            self.turns = (torch.arange(self.N, device=device) % 2).to(torch.int8)
        else:
            self.turns.zero_()

        # First deal: exactly three draws
        for _ in range(3):
            self.draw()

    def compare_cards(self, idx1, idx2):
        """Return winners (0 if player 1 wins, 1 if player 2 wins) for each game."""
        seme1, num1 = idx1 // 10, idx1 % 10
        seme2, num2 = idx2 // 10, idx2 % 10

        val1 = VALUES_TENSOR[num1]
        val2 = VALUES_TENSOR[num2]

        same_suit = seme1 == seme2
        diff_suit = ~same_suit
        seme1_briscola = seme1 == self.briscole
        seme2_briscola = seme2 == self.briscole

        winners = torch.zeros_like(idx1, device=device, dtype=torch.int8)  # 0 if P1 wins, 1 if P2 wins
        winners[diff_suit & seme2_briscola] = 1
        winners[same_suit & (val2 > val1)] = 1
        winners[same_suit & (val2 == val1) & (num2 > num1)] = 1
        winners[diff_suit & ~seme1_briscola & (self.turns == 1)] = 1
        return winners

    def count_points(self):
        """Returns the players' current point counts (per game)."""
        points = (self.taken * VALUES_TENSOR).sum(dim=2)  # (2, N)
        return points[0], points[1]  # (N,), (N,)

    def check_finished(self):
        """Returns whether the game (first table) is over (compat with original behavior)."""
        return (self.hands[0, 0].sum().item() == 0) and (self.hands[1, 0].sum().item() == 0)

    def check_winners(self):
        """Returns points and winner per game (0/1/2 for draw)."""
        count1, count2 = self.count_points()
        winners = torch.zeros_like(count1, dtype=torch.int8, device=device)
        winners.masked_fill_(count2 > count1, 1)
        winners.masked_fill_(count2 == count1, 2)
        return count1, count2, winners

    def draw(self):
        """All decks draw in lockstep: two cards per game, vectorized with scatter_."""
        if self.deck_pos is None or self.deck_pos < 1:
            return

        # Last two cards across every deck (N,)
        c_first  = self.decks[:, self.deck_pos]
        c_second = self.decks[:, self.deck_pos - 1]

        # Per-player card indices (N,1)
        turn = self.turns
        p0_idx = torch.where(turn == 0, c_first,  c_second).unsqueeze(1)
        p1_idx = torch.where(turn == 0, c_second, c_first ).unsqueeze(1)

        ones = torch.ones((self.N, 1), dtype=torch.float32, device=device)

        # Scatter into (2, N, 40) — each player plane is (N, 40)
        self.hands[0].scatter_(1, p0_idx, ones)
        self.hands[1].scatter_(1, p1_idx, ones)

        # Advance shared pointer by two
        self.deck_pos -= 2
