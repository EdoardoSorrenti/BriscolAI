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

class Player:
    def __init__(self):
        self.hand = []
        self.taken = []

    def randmove(self, othercard = None):
        """Produce valid randomized moves."""
        if len(self.hand) <= 1:
            return 0
        return random.randint(0, len(self.hand)-1)

    
    def manualmove(self, othercard = None):
        """Allow for basic interaction with the game through CLI."""
        if othercard:
            print(f"L'avversario ha giocato un {othercard[1]} di {SEED_NAMES[othercard[0]]}")
        print(f"Hai a disposizione: {self.hand}, scegli la carta (0-2)")
        carta = int(input())
        return carta
    
    def count_points(self):
        """Returns the player's current point count."""
        points = 0
        for card in self.taken:
            points += VALUES[card[1]]
        return points

class Game:
    @staticmethod
    def simulationInit():
        """Returns a Game instance with a randomly shuffled deck."""
        game = Game()
        game.players = [Player(), Player()]
        game.deck = copy.deepcopy(DECK)
        random.shuffle(game.deck)
        for x in range(3):
            game.players[0].hand.append(game.deck.pop(0))
            game.players[1].hand.append(game.deck.pop(0))
        
        game.briscola = game.deck[-1]
        game.on_table = [None, None]
        game.turno = 0
        return game
    
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

    def check_finished(self):
        """Returns wether the game is over."""
        if not self.players[0].hand and not self.players[1].hand:
            return True
        else:
            return False

    def check_winner(self):
        """Returns index of the player with currently more points, -1 if draw."""
        count1 = self.players[0].count_points()
        count2 = self.players[1].count_points()
        if count1 > count2:
            return 0
        elif count2 > count1: 
            return 1
        else:
            return -1

    def draw(self):
        """If available makes all players draw a card."""
        if len(self.deck):
                self.players[self.turno].hand.append(self.deck.pop(0))
                self.players[1-self.turno].hand.append(self.deck.pop(0))


