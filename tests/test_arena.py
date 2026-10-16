import numpy as np

from src import arena


def test_correctness_shape(fitted, small_data):
    _, x_te, _, y_te = small_data
    names, correct, preds = arena.correctness(fitted, x_te, y_te)
    assert correct.shape == (len(x_te), len(names))
    assert correct.dtype == bool
    assert set(preds) == set(names)


def test_games_never_pair_a_model_with_itself():
    correct = np.random.default_rng(0).random((50, 4)) > 0.5
    games, _ = arena.sample_games(correct, 5000, seed=1)
    assert (games[:, 0] != games[:, 1]).all()
    assert games[:, :2].max() == 3 and games[:, 2].max() < 50


def test_game_scores_follow_correctness():
    correct = np.array([[True, False], [False, True], [True, True], [False, False]])
    games, score = arena.sample_games(correct, 400, seed=2)
    for (a, b, d), s in zip(games, score):
        ca, cb = correct[d, a], correct[d, b]
        assert s == (0.5 if ca == cb else float(ca))


def test_record_counts_add_up():
    correct = np.random.default_rng(3).random((80, 3)) > 0.4
    games, score = arena.sample_games(correct, 900, seed=4)
    w, l, t = arena.record(games, score, 3)
    assert w.sum() == l.sum() == int((score != 0.5).sum())
    assert (w + l + t).sum() == 2 * len(score)
