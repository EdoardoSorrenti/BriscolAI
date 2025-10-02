from config import *
import random, copy


# Creazione mazzo base
DECK = []
for seed in SEEDS:
    for num in range(1,11):
        DECK.append((seed, num))

# Rappresentazione in testo di una carta
def reprCard(card):
    return f"{card[1]} di {SEED_NAMES[card[0]]}"

class Player:
    def __init__(self):
        self.hand = []
        self.taken = []

    # produce mosse casuali valide
    def randmove(self, othercard = None):
        if len(self.hand) <= 1:
            return 0
        return random.randint(0, len(self.hand)-1)

    def manualmove(self, othercard = None):
        if othercard:
            print(f"L'avversario ha giocato un {othercard[1]} di {SEED_NAMES[othercard[0]]}")
        print(f"Hai a disposizione: {self.hand}, scegli la carta (0-2)")
        carta = int(input())
        return carta
    
    def count_points(self):
        points = 0
        for card in self.taken:
            points += VALUES[card[1]]
        return points

class Game:
    @staticmethod
    def simulationInit():
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
        if not self.players[0].hand and not self.players[1].hand:
            return True
        else:
            return False

    def check_winner(self):
        count1 = self.players[0].count_points()
        count2 = self.players[1].count_points()
        if count1 > count2:
            return 0
        elif count2 > count1: 
            return 1
        else:
            return -1

    def draw(self):
        if len(self.deck):
                self.players[self.turno].hand.append(self.deck.pop(0))
                self.players[1-self.turno].hand.append(self.deck.pop(0))

    def mainloop(self):
        while not self.check_finished():
            player1 = self.players[0]
            player2 = self.players[1]
            if self.turno == 0:
                card1 = player1.hand.pop(player1.randmove())
                card2 = player2.hand.pop(player2.randmove(card1))
            else:
                card2 = player2.hand.pop(player2.randmove())
                card1 = player1.hand.pop(player1.randmove(card2))
            winner = self.compare_hands(card1, card2)

            self.turno = winner
            self.draw()

            self.players[winner].taken += (card1, card2)
            
                
        return self.check_winner()

if __name__ == "__main__":           
    winner1 = 0
    winner2 = 0
    draws = 0
    for x in range(10000):
        game = Game.simulationInit()
        winner = game.mainloop()
        if winner== 0:
            winner1 += 1
        elif winner == 1:
            winner2 += 1
        else:
            draws += 1
    print(f"Winner 1: {winner1}, Winner 2: {winner2}, draws: {draws}")
