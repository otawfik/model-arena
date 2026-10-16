"""Game engine: per-document head-to-head games between fitted models.

A game is one held-out document and one pair of models, the way a Chatbot
Arena game is one prompt and two chatbots. The model that alone gets the label
right wins; both right or both wrong is a tie. Sampling many games gives a
large, independent set of outcomes to rate, instead of replaying the same
whole-test-set comparison over and over (which is deterministic, so every
replay carries no new information and Elo just drifts with match order).
"""
import numpy as np


def correctness(models, x_test, y_test):
    """Predict once per model. Returns (names, correct[n_docs, n_models], preds)."""
    names = list(models)
    y = np.asarray(y_test)
    preds = {n: np.asarray(models[n].predict(x_test)) for n in names}
    correct = np.column_stack([preds[n] == y for n in names])
    return names, correct, preds


def sample_games(correct, n_games, seed=7):
    """Sample random (doc, model pair) games from a correctness matrix.

    Returns an int array [n_games, 3] of (model_a, model_b, doc) and a float
    array of scores for model_a: 1 win, 0 loss, 0.5 tie.
    """
    rng = np.random.default_rng(seed)
    n_docs, n_models = correct.shape
    a = rng.integers(0, n_models, n_games)
    b = (a + rng.integers(1, n_models, n_games)) % n_models  # b != a, uniform
    doc = rng.integers(0, n_docs, n_games)
    ca, cb = correct[doc, a], correct[doc, b]
    score = np.where(ca == cb, 0.5, ca.astype(float))
    return np.column_stack([a, b, doc]), score


def record(games, score, n_models):
    """Per-model (wins, losses, ties) over a set of games."""
    w = np.zeros(n_models, int)
    l = np.zeros(n_models, int)
    t = np.zeros(n_models, int)
    a, b = games[:, 0], games[:, 1]
    np.add.at(w, a[score == 1], 1)
    np.add.at(l, b[score == 1], 1)
    np.add.at(w, b[score == 0], 1)
    np.add.at(l, a[score == 0], 1)
    np.add.at(t, a[score == 0.5], 1)
    np.add.at(t, b[score == 0.5], 1)
    return w, l, t
