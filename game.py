import random, copy

SEEDS = ("B", "S", "C", "D")
SEED_NAMES = {"B":"Bastoni", "S":"Spade", "C":"Coppe", "D":"Denari"}
VALUES = {1:11, 2:0, 3:10, 4:0, 5:0, 6:0, 7:0, 8:2, 9:3, 10:4} # Valori carte a briscola

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
        return game
    
    def compare_hands(self, card1, card2):
        seme1, num1 = card1[0], card1[1]
        seme2, num2 = card2[0], card2[1]
        briscola = self.briscola[0]
        if seme1 != seme2 and seme2 != briscola:
            return 0
        elif seme1 != seme2 and seme2 == briscola:
            return 1
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

    def mainloop(self):
        turno = 0
        while len(self.players[turno].hand):
            player1 = self.players[turno]
            player2 = self.players[1-turno]
            move1 = player1.randmove()
            if move1 >= len(player1.hand):
                return 1
            card1 = player1.hand.pop(move1)
            move2 = player2.randmove(card1)
            if move2 >= len(player2.hand):
                return 0
            card2 = player2.hand.pop(move2)
            relative_hand_winner = self.compare_hands(card1, card2)
            if turno == 0: 
                hand_winner = relative_hand_winner
            else:
                hand_winner = 1 - relative_hand_winner

            turno = hand_winner

            self.players[hand_winner].taken += (card1, card2)
            if len(self.deck) :
                self.players[hand_winner].hand.append(self.deck.pop(0))
                self.players[1-hand_winner].hand.append(self.deck.pop(0))
                
        count1 = self.players[0].count_points()
        count2 = self.players[1].count_points()
        if count1 > count2:
            return 0
        elif count2 > count1: 
            return 1
        else:
            return -1
            
winner1 = 0
winner2 = 0
draws = 0
for x in range(100000):
    game = Game.simulationInit()
    winner = game.mainloop()
    if winner== 0:
        winner1 += 1
    elif winner == 1:
        winner2 += 1
    else:
        draws += 1
print(f"Winner 1: {winner1}, Winner 2: {winner2}, draws: {draws}")

        