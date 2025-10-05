"""
Testing script for the Game engine
Runs ITERATIONS randomly played games
Results should roughly be: P1 52.7%; P2 45.6%; DRAW: 1.7%
"""

import random
from game import Game
from time import time

ITERATIONS = 100_000

def randmove(hand):
    """Produce valid randomized moves."""
    if len(hand) <= 1:
        return 0
    return random.randint(0, len(hand)-1)


def randloop(session):
    """Plays a whole game with both players playing random moves."""
    while not session.check_finished():
        if session.turno == 0:
            card1 = session.hands[0].pop(randmove(session.hands[0]))
            card2 = session.hands[1].pop(randmove(session.hands[1]))
        else:
            card2 = session.hands[1].pop(randmove(session.hands[1]))
            card1 = session.hands[0].pop(randmove(session.hands[0]))
        winner = session.compare_hands(card1, card2)
        session.turno = winner
        session.draw()
        session.taken[winner] += (card1, card2)

    return session.check_winner()

def testloops(n_its):
    """Runs n_its iterations and prints the result and average speed"""
    winner1 = 0
    winner2 = 0
    draws = 0
    start_time = time()
    game = Game()
    for x in range(n_its):
        game.reset()
        winner = randloop(game)
        if winner== 0:
            winner1 += 1
        elif winner == 1:
            winner2 += 1
        else:
            draws += 1
    end_time = time()
    tot_time = end_time - start_time
    games_per_second = n_its / tot_time
    print(f"Winner 1: {winner1/n_its*100:.1f}%, Winner 2: {winner2/n_its*100:.1f}%, draws: {draws/n_its*100:.1f}%")
    print(f"Compute speed: {round(games_per_second)} games/second")

if __name__ == "__main__":
    testloops(ITERATIONS)