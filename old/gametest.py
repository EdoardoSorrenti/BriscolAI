"""
Testing script for the Game engine
Runs ITERATIONS randomly played games
Results should roughly be: P1 52.7%; P2 45.6%; DRAW: 1.7%
"""

from tensorgame import Game
from time import perf_counter

ITERATIONS = 10_000

def fastloop(session):
    """Plays a whole game with both players playing random moves."""
    while not session.check_finished():
        card1 = session.hands[0].argmax().item()
        session.hands[0][card1] = 0
        card2 = session.hands[1].argmax().item()
        session.hands[1][card2] = 0
        winner = session.compare_hands(card1, card2)
        session.turno = winner
        session.draw()
        session.taken[winner][card1] = 1
        session.taken[winner][card2] = 1

    return session.check_winner()

def testloops(n_its):
    """Runs n_its iterations and prints the result and average speed"""
    start_time = perf_counter()
    game = Game()
    outcomes = [0, 0, 0]  # Wins for player 1, player 2, draws
    for x in range(n_its):
        game.reset()
        winner = fastloop(game)
        outcomes[winner] += 1
    end_time = perf_counter()
    tot_time = end_time - start_time
    games_per_second = n_its / tot_time
    print(f"Winner 1: {outcomes[0]/n_its*100:.1f}%, Winner 2: {outcomes[1]/n_its*100:.1f}%, draws: {outcomes[2]/n_its*100:.1f}%")
    print(f"Compute speed: {round(games_per_second)} games/second")

if __name__ == "__main__":
    testloops(ITERATIONS)