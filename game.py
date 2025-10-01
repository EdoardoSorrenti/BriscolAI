import random, copy

SEEDS = ("Bastoni", "Spade", "Coppe", "Denari")

DECK = []
for seed in SEEDS:
    for num in range(1,11):
        DECK.append((seed, num))

print()

class Player:
    def __init__(self):
        self.hand = []
        self.taken = []

    def manualmove(self, othercard = None):
        if othercard:
            print(f"L'avversario ha giocato un {othercard[1]} di {othercard[0]}")
        print(f"Hai a disposizione: {self.hand}, scegli la carta (0-2)")
        carta = int(input())
        return carta
        

class Game:
    def simulationInit():
        game = Game()
        game.players = [Player(), Player()]
        game.deck = copy.deepcopy(DECK)
        random.shuffle(game.deck)
        for x in range(3):
            game.players[0].hand.append(game.deck.pop(0))
            game.players[1].hand.append(game.deck.pop(0))
        
        game.briscola = game.deck[-1]
        game.cardsplayed = []
        return game
    
    def mainloop(self):
        turno = 0
        finishedGame = False
        while not finishedGame:
            player1 = self.player[turno]
            player2 = self.player[1-turno]
            move1 = player1.manualmove()
            card1 = player1.hand[move1]
            move2 = player2.manualmove(player1.hand[move1])
            card2 = player2.hand[move2]
            


    

game = Game.simulationInit()
print(f"Mano giocatore 1: {game.players[0].hand}")
print(f"Mano giocatore 2: {game.players[1].hand}")
print(f"Briscola: {game.briscola}")
print(game.deck)

        