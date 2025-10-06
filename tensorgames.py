import random, copy
import torch
from config import device_name

device = torch.device(device_name)

torch.set_default_device(device)

# Point values assigned to each card
VALUES = {0:11, 1:0, 2:10, 3:0, 4:0, 5:0, 6:0, 7:2, 8:3, 9:4} # Valori carte a briscola

VALUES_TENSOR = torch.tensor([11,0,10,0,0,0,0,2,3,4]*4, dtype=torch.float32, device=device)

# Creazione mazzo base
DECK = list(range(40)) # 1-40

class Games:
    def __init__(self, N, alternate_turns=True):
        """Initializes tensors for an empty game."""
        self.N = N
        self.alternate_turns = alternate_turns

        # NN inputs
        self.hands = [torch.zeros((N, 40), dtype=torch.float32, device=device), torch.zeros((N,40), dtype=torch.float32, device=device)] # Carte in mano
        self.taken = [torch.zeros((N,40), dtype=torch.float32, device=device), torch.zeros((N,40), dtype=torch.float32, device=device)] # Carte prese
        self.briscole_cards = torch.zeros((N,40), dtype=torch.float32, device=device) # Carta di briscola
        self.on_table = torch.zeros((N,40), dtype=torch.float32, device=device) # Carte sul tavolo

        # Utils
        self.turns = torch.zeros(N, dtype=torch.int8, device=device)
        self.briscole = torch.zeros(N, dtype=torch.int8, device=device)

    
    def reset(self, turno=0):
        """Resets the games to initial state."""
        self.hands[0].zero_()
        self.hands[1].zero_()
        self.taken[0].zero_()
        self.taken[1].zero_()
        self.briscole_cards.zero_()
        self.on_table.zero_()

        self.briscole.zero_()
        self.turns.zero_()

        self.decks = [random.sample(DECK, len(DECK)) for _ in range(self.N)]

        for n in range(self.N):
            for _ in range(3):
                self.hands[0][n, self.decks[n].pop()] = 1
                self.hands[1][n, self.decks[n].pop()] = 1

            self.briscole[n] = self.decks[n][0]//10
            self.briscole_cards[n, self.briscole[n]] = 1

            if self.alternate_turns:
                self.turns[n] = n%2 
            else:
                self.turns[n] = 0
    
    
    def compare_cards(self, idx1, idx2):
        turns = self.turns

        seme1, num1 = idx1 // 10, idx1 % 10
        seme2, num2 = idx2 // 10, idx2 % 10

        val1 = VALUES_TENSOR[num1]
        val2 = VALUES_TENSOR[num2]

        same_suit = seme1 == seme2
        diff_suit = ~same_suit
        seme1_briscola = seme1 == self.briscole
        seme2_briscola = seme2 == self.briscole
        
        winners = torch.zeros_like(idx1, device=device, dtype=torch.int8) # 0 if player 1 wins, 1 if player 2 wins

        winners[diff_suit & seme2_briscola] = 1
        winners[same_suit & (val2 > val1)] = 1
        winners[same_suit & (val2 == val1) & (num2 > num1)] = 1
        winners[diff_suit & ~seme1_briscola & turns == 1] = 1

        return winners

    def count_points(self):
        """Returns the player's current point count."""
        points1 = (self.taken[0]*VALUES_TENSOR).sum(dim=1)
        points2 = (self.taken[1]*VALUES_TENSOR).sum(dim=1)
        return points1, points2

    def check_finished(self):
        """Returns wether the game is over."""
        if not self.hands[0][0].sum() and not self.hands[1][0].sum():
            return True
        else:
            return False

    def check_winners(self):
        """Returns index of the player with currently more points, 2 if draw."""
        count1, count2 = self.count_points()
        winners = torch.zeros_like(count1, dtype=torch.int8, device=device)

        winners[count2>count1] = 1
        winners[count2 == count1] = 2


        return winners

    def draw(self):
        """If available makes all players draw a card."""
        for i, deck in enumerate(self.decks):
            if len(deck):
                self.hands[self.turns[i]][i][deck.pop()] = 1
                self.hands[1-self.turns[i]][i][deck.pop()] = 1