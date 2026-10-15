"""ELO rating system, chess-style, for pairwise model matchups.

Each model starts at 1200. After a head-to-head battle the winner takes
points from the loser proportional to the upset: beating a stronger model
earns more. score_a is 1.0 for an A win, 0.0 for a B win, 0.5 for a tie.
"""


def expected_score(r_a, r_b):
    """Probability that A beats B given current ratings."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def update_ratings(r_a, r_b, score_a, k=32):
    """Return (new_a, new_b) after a single matchup."""
    e_a = expected_score(r_a, r_b)
    new_a = r_a + k * (score_a - e_a)
    new_b = r_b + k * ((1.0 - score_a) - (1.0 - e_a))
    return round(new_a, 2), round(new_b, 2)
