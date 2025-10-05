import random, copy
import torch

# Point values assigned to each card
VALUES = {0:11, 1:0, 2:10, 3:0, 4:0, 5:0, 6:0, 7:2, 8:3, 9:4} # Valori carte a briscola

VALUES_TENSOR = torch.tensor([11,0,10,0,0,0,0,2,3,4]*4, dtype=torch.float32)

# Creazione mazzo base
DECK = list(range(40)) # 1-40

class Game:
    def __init__(self):
        """Initializes tensors for an empty game."""
        self.hands = [torch.zeros(40, dtype=torch.float32), torch.zeros(40, dtype=torch.float32)] # Carte in mano
        self.taken = [torch.zeros(40, dtype=torch.float32), torch.zeros(40, dtype=torch.float32)] # Carte prese
        self.briscola_card = torch.zeros(40, dtype=torch.float32) # Carta di briscola

        self.on_table = torch.zeros(40, dtype=torch.float32) # Carte sul tavolo
    
    def reset(self, turno=0):
        """Resets the game to initial state."""
        self.hands[0].zero_()
        self.hands[1].zero_()
        self.taken[0].zero_()
        self.taken[1].zero_()
        self.briscola_card.zero_()
        self.on_table.zero_()

        self.deck = copy.copy(DECK)
        random.shuffle(self.deck)
        for _ in range(3):
            self.hands[0][self.deck.pop()] = 1
            self.hands[1][self.deck.pop()] = 1

        self.briscola = self.deck[0]
        self.briscola_card[self.briscola] = 1
        self.turno = turno  # Player to play first, 0 or 1
    
    def compare_hands(self, card1, card2):
        """Compares cards played by P1 and P2,
        Order is important: card1 is played by P1 and card2 by P2,
        Returns index of the winning card (0 or 1)"""
        seme1, num1 = card1//10, card1%10
        seme2, num2 = card2//10, card2%10
        briscola = self.briscola//10
        if seme1 != seme2:
            if seme1 == briscola:
                return 0
            elif seme2 == briscola:
                return 1
            else:
                return self.turno
        elif seme1 == seme2:
            if VALUES[num1] > VALUES[num2]:
                return 0
            elif VALUES[num1] == VALUES[num2]:
                if num1>num2:
                    return 0
                else:
                    return 1
            else:
                return 1
            
    def count_points(self):
        """Returns the player's current point count."""
        points1 = (self.taken[0]*VALUES_TENSOR).sum()
        points2 = (self.taken[1]*VALUES_TENSOR).sum()
        return points1, points2

    def check_finished(self):
        """Returns wether the game is over."""
        if not self.hands[0].sum() and not self.hands[1].sum():
            return True
        else:
            return False

    def check_winner(self):
        """Returns index of the player with currently more points, 2 if draw."""
        count1, count2 = self.count_points()
        if count1 > count2:
            return 0
        elif count2 > count1: 
            return 1
        else:
            return 2

    def draw(self):
        """If available makes all players draw a card."""
        if len(self.deck):
            self.hands[self.turno][self.deck.pop()] = 1
            self.hands[1-self.turno][self.deck.pop()] = 1