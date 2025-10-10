import torch


# Point values assigned to each card
VALUES = {0:11, 1:0, 2:10, 3:0, 4:0, 5:0, 6:0, 7:2, 8:3, 9:4}  # Valori carte a briscola
VALUES_TENSOR = torch.tensor([11,0,10,0,0,0,0,2,3,4]*4)

# Creazione mazzo base
DECK = list(range(40))  # 0..39

class Games:
    def __init__(self, N, alternate_turns=True):
        """Initializes tensors for an empty game."""
        self.N = N
        self.alternate_turns = alternate_turns

        # Vectorized: (2, N, 40) so hands[0] -> player 1, hands[1] -> player 2
        self.hands  = torch.zeros((2, N, 40))  # Carte in mano
        self.taken  = torch.zeros((2, N, 40))  # Carte prese
        self.briscole_cards = torch.zeros((N, 40))
        self.on_table = torch.zeros((N, 40))

        device = self.hands.device
        dtype = self.hands.dtype
        self.ones = torch.ones((N, 1), device=device, dtype=dtype)
        self.values_fp32 = VALUES_TENSOR.to(device=device, dtype=torch.float32)
        self.rand_buffer = torch.empty((N, 40), device=device, dtype=torch.float32)

        # Utils
        self.player2_starts = torch.zeros(N, dtype=torch.bool)     # 0 -> P0 first, 1 -> P1 first
        self.briscole = torch.zeros(N, dtype=torch.int8)  # suit (0..3)
        self.trick_winners = torch.zeros(N, dtype=torch.int8)  # 0/1 per ogni mano

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
        self.player2_starts.zero_()

        # Create tensorized decks: each row a shuffled permutation of 0..39.
        uniforms = torch.rand((self.N, 40))
        self.decks = uniforms.argsort(dim=1)  # (N, 40), int64
        self.deck_pos = 39  # last index (shared across all rows)

        # Briscola suit determined by the "bottom" card (index 0)
        self.briscole = (self.decks[:, 0] // 10).to(dtype=torch.int8)

        # One-hot suits in briscole_cards via scatter_ (indices must be long)
        suit_idx = self.briscole.to(torch.long).unsqueeze(1)             # (N,1)
        self.briscole_cards.zero_()
        self.briscole_cards.scatter_(1, suit_idx, torch.ones((self.N,1)))

        # Initial turns
        if self.alternate_turns:
            self.player2_starts = (torch.arange(self.N) % 2).bool()
        else:
            self.player2_starts.zero_()

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

        self.trick_winners.zero_()  # 0 if P1 wins, 1 if P2 wins
        self.trick_winners[diff_suit & seme2_briscola] = 1 # P2 briscola, P1 not
        self.trick_winners[same_suit & (val2 > val1)] = 1 # same suit, higher value
        self.trick_winners[same_suit & (val2 == val1) & (num2 > num1)] = 1 # same suit, same value, higher rank
        self.trick_winners[diff_suit & ~seme1_briscola & self.player2_starts] = 1 # P2 leads, neither briscola

        self.player2_starts.copy_(self.trick_winners.bool())

        return self.trick_winners

    def count_points(self):
        """Returns the players' current point counts (per game)."""
        taken_fp32 = self.taken.to(torch.float32)
        points = (taken_fp32 * self.values_fp32).sum(dim=2)  # (2, N)
        return points[0], points[1]  # (N,), (N,)

    def check_winners(self):
        """Returns points and winner per game (0/1/2 for draw)."""
        count1, count2 = self.count_points()
        winners = torch.zeros_like(count1, dtype=torch.int8)
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
        turn = self.player2_starts
        p0_idx = torch.where(turn == 0, c_first,  c_second).unsqueeze(1)
        p1_idx = torch.where(turn == 0, c_second, c_first ).unsqueeze(1)

        # Scatter into (2, N, 40) — each player plane is (N, 40)
        self.hands[0].scatter_(1, p0_idx, self.ones)
        self.hands[1].scatter_(1, p1_idx, self.ones)


        # Advance shared pointer by two
        self.deck_pos -= 2

    # --- Helper utilities for simulation control ---

    def split_turns(self):
        """Returns (mask, indices) tuples for games where player 0 or 1 acts next."""
        turn0_mask = ~self.player2_starts
        turn1_mask = self.player2_starts
        turn0_idx = torch.nonzero(turn0_mask, as_tuple=True)[0]
        turn1_idx = torch.nonzero(turn1_mask, as_tuple=True)[0]
        return (turn0_mask, turn0_idx), (turn1_mask, turn1_idx)

    def hand_subset(self, player_id, indices=None):
        """Returns hands for the specified player, optionally restricted to indices."""
        if indices is None:
            return self.hands[player_id]
        return self.hands[player_id].index_select(0, indices)

    def place_on_table(self, game_indices, card_indices):
        """Marks the provided cards as currently on the table for selected games."""
        self.on_table[game_indices, card_indices] = 1

    def remove_played_cards(self, card_player0, card_player1):
        """Removes the cards just played from both players' hands."""
        self.hands[0].scatter_(1, card_player0.unsqueeze(1), 0)
        self.hands[1].scatter_(1, card_player1.unsqueeze(1), 0)

    def record_taken_cards(self, winners, card_player0, card_player1):
        """Updates taken piles according to the trick winners."""
        p0_idx = torch.nonzero(winners == 0, as_tuple=True)[0]
        if p0_idx.numel():
            self.taken[0][p0_idx, card_player0[p0_idx]] = 1
            self.taken[0][p0_idx, card_player1[p0_idx]] = 1

        p1_idx = torch.nonzero(winners == 1, as_tuple=True)[0]
        if p1_idx.numel():
            self.taken[1][p1_idx, card_player0[p1_idx]] = 1
            self.taken[1][p1_idx, card_player1[p1_idx]] = 1

    def clear_table(self):
        """Resets table state once the trick is resolved."""
        self.on_table.zero_()

    def refresh_random_buffer(self):
        """Refills the cached random scores used for stochastic moves."""
        self.rand_buffer.uniform_()

    def pick_random_cards(self, player_id, indices):
        """Returns random valid card indices for the given player and subset of games."""
        if indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.hands.device)

        scores = self.rand_buffer.index_select(0, indices)
        hands = self.hand_subset(player_id, indices).to(scores.dtype)
        scores.mul_(hands)
        return scores.argmax(dim=1)
