import numpy as np

from src import ratings


def simulate(strengths, n_games, seed=0):
    """Games drawn from a known Bradley-Terry model (no ties)."""
    rng = np.random.default_rng(seed)
    m = len(strengths)
    a = rng.integers(0, m, n_games)
    b = (a + rng.integers(1, m, n_games)) % m
    p = 1 / (1 + 10 ** ((strengths[b] - strengths[a]) / 400))
    score = (rng.random(n_games) < p).astype(float)
    return np.column_stack([a, b, np.zeros(n_games, int)]), score


def test_bradley_terry_recovers_known_strengths():
    true = np.array([1100.0, 1000.0, 900.0])
    games, score = simulate(true, 40000)
    W, N = ratings.pair_counts(games, score, 3)
    fit = ratings.bradley_terry(W, N)
    assert np.allclose(fit - fit.mean(), true - true.mean(), atol=12)


def test_bradley_terry_centred_on_base():
    games, score = simulate(np.array([1200.0, 1000.0]), 2000)
    W, N = ratings.pair_counts(games, score, 2)
    assert abs(ratings.bradley_terry(W, N).mean() - ratings.BASE) < 1e-6


def test_bradley_terry_ignores_game_order():
    games, score = simulate(np.array([1100.0, 1000.0, 950.0]), 3000)
    idx = np.random.default_rng(5).permutation(len(score))
    a = ratings.bradley_terry(*ratings.pair_counts(games, score, 3))
    b = ratings.bradley_terry(*ratings.pair_counts(games[idx], score[idx], 3))
    assert np.allclose(a, b)


def test_online_elo_depends_on_order():
    games, score = simulate(np.array([1050.0, 1000.0, 950.0]), 3000)
    assert ratings.order_spread(games, score, 3, n_orders=10, k=16).max() > 1.0


def test_win_probability_is_complementary():
    p = ratings.win_probability(np.array([1100.0, 1000.0, 1000.0]))
    assert np.allclose(p + p.T, 1.0)
    assert abs(p[1, 2] - 0.5) < 1e-9
    assert abs(p[0, 1] - 1 / (1 + 10 ** (-100 / 400))) < 1e-9


def test_bootstrap_interval_contains_point_estimate():
    games, score = simulate(np.array([1100.0, 1000.0, 900.0]), 5000)
    point = ratings.bradley_terry(*ratings.pair_counts(games, score, 3))
    boot, ranks = ratings.bootstrap(games, score, 3, n_boot=100)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    assert ((lo <= point) & (point <= hi)).all()
    assert sorted(ranks[0]) == [1, 2, 3]
