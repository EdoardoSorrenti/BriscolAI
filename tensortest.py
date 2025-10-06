"""
Testing script for the Game engine
Runs ITERATIONS randomly played games
Results should roughly be: P1 52.7%; P2 45.6%; DRAW: 1.7%
"""

from tensorgames import Games
from time import perf_counter
import torch

from random import choice

ITERATIONS = 10_000

def random_pick_multihot(x):
    # Random numbers, set zeros where x is zero
    rand = torch.rand_like(x, dtype=torch.float) * x
    # Pick the argmax of random numbers (guaranteed to pick one of the nonzero entries)
    return rand.argmax(dim=1)

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

def tensorloop(games):
    game_indices = torch.arange(games.N, dtype=torch.int32)
    while not games.check_finished():
        card1 = random_pick_multihot(games.hands[0])
        games.hands[0][game_indices, card1] = 0
        games.on_table[game_indices, card1] = 1

        card2 = random_pick_multihot(games.hands[1])
        games.hands[1][game_indices, card2] = 0
        games.on_table[game_indices, card2] = 1
        winners = games.compare_cards(card1, card2)

        p0_wins_mask = winners == 0
        p1_wins_mask = winners == 1

        # Use the masks to update the 'taken' tensor for player 0
        games.taken[0][game_indices[p0_wins_mask], card1[p0_wins_mask]] = 1
        games.taken[0][game_indices[p0_wins_mask], card2[p0_wins_mask]] = 1

        # Use the masks to update the 'taken' tensor for player 1
        games.taken[1][game_indices[p1_wins_mask], card1[p1_wins_mask]] = 1
        games.taken[1][game_indices[p1_wins_mask], card2[p1_wins_mask]] = 1

        games.turns = winners   

        games.draw()
    
    winners_array = games.check_winners()
    winners = [(winners_array==0).sum(),(winners_array==1).sum(),(winners_array==2).sum()]
    return winners

def testloops(n_its):
    """Runs n_its iterations and prints the result and average speed"""
    start_time = perf_counter()
    games = Games(n_its, alternate_turns=False)
    games.reset()
    outcomes = tensorloop(games)
    end_time = perf_counter()
    tot_time = end_time - start_time
    games_per_second = n_its / tot_time
    print(f"Winner 1: {outcomes[0]/n_its*100:.1f}%, Winner 2: {outcomes[1]/n_its*100:.1f}%, draws: {outcomes[2]/n_its*100:.1f}%")
    print(f"Compute speed: {round(games_per_second)} games/second")

if __name__ == "__main__":
    testloops(ITERATIONS)