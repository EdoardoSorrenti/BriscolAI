import random, copy

# Card seeds by initial (to save memory) and full names
SEEDS = ("B", "S", "C", "D") 
SEED_NAMES = {"B":"Bastoni", "S":"Spade", "C":"Coppe", "D":"Denari"} 

# Point values assigned to each card
VALUES = {1:11, 2:0, 3:10, 4:0, 5:0, 6:0, 7:0, 8:2, 9:3, 10:4} # Valori carte a briscola

# Creazione mazzo base
DECK = []
for seed in SEEDS:
    for num in range(1,11):
        DECK.append((seed, num))    

class Game:
    
    def reset(self):
        """Resets the game to initial state."""
        self.hands = [[], []]
        self.taken = [[], []]
        self.deck = copy.deepcopy(DECK)
        random.shuffle(self.deck)
        for x in range(3):
            self.hands[0].append(self.deck.pop(0))
            self.hands[1].append(self.deck.pop(0))

        self.briscola = self.deck[-1]
        self.on_table = [None, None]
        self.turno = 0
    
    def compare_hands(self, card1, card2):
        """Compares cards played by P1 and P2,
        Order is important: card1 is played by P1 and card2 by P2,
        Returns index of the winning card (0 or 1)"""
        seme1, num1 = card1[0], card1[1]
        seme2, num2 = card2[0], card2[1]
        briscola = self.briscola[0]
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
        points1 = 0
        for card in self.taken[0]:
            points1 += VALUES[card[1]]
        points2 = 0
        for card in self.taken[1]:
            points2 += VALUES[card[1]]
        return points1, points2

    def check_finished(self):
        """Returns wether the game is over."""
        if not self.hands[0] and not self.hands[1]:
            return True
        else:
            return False

    def check_winner(self):
        """Returns index of the player with currently more points, -1 if draw."""
        count1, count2 = self.count_points()
        if count1 > count2:
            return 0
        elif count2 > count1: 
            return 1
        else:
            return -1

    def draw(self):
        """If available makes all players draw a card."""
        if len(self.deck):
                self.hands[self.turno].append(self.deck.pop(0))
                self.hands[1-self.turno].append(self.deck.pop(0))