import torch
from pathlib import Path

VALUES_PER_RANK = torch.tensor([11, 0, 10, 0, 0, 0, 0, 2, 3, 4], dtype=torch.int32)
VALUES_PER_CARD = VALUES_PER_RANK.repeat(4)


def _winner_for_cards(card1: int, card2: int, briscola: int, player2_starts: int) -> int:
    suit1, rank1 = divmod(card1, 10)
    suit2, rank2 = divmod(card2, 10)
    val1 = int(VALUES_PER_CARD[card1])
    val2 = int(VALUES_PER_CARD[card2])

    if suit1 != suit2:
        is_c1_briscola = suit1 == briscola
        is_c2_briscola = suit2 == briscola
        if is_c2_briscola and not is_c1_briscola:
            return 1
        if is_c1_briscola and not is_c2_briscola:
            return 0
        if player2_starts and not is_c1_briscola:
            return 1
        return 0

    if val2 > val1:
        return 1
    if val2 < val1:
        return 0
    if rank2 > rank1:
        return 1
    return 0


def build_winner_table() -> torch.Tensor:
    table = torch.zeros((4, 2, 40, 40), dtype=torch.int8)
    for briscola in range(4):
        for p2_start in range(2):
            for card1 in range(40):
                for card2 in range(40):
                    table[briscola, p2_start, card1, card2] = _winner_for_cards(
                        card1, card2, briscola, p2_start
                    )
    return table


def main() -> None:
    table = build_winner_table()
    output_path = Path(__file__).resolve().with_name("lookup_tables.pt")
    torch.save({"winner": table}, output_path)
    print(f"Saved lookup table to {output_path}")


if __name__ == "__main__":
    main()
