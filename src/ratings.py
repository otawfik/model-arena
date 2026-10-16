"""Ratings from game outcomes: online Elo vs Bradley-Terry.

Online Elo updates after every game, so the final number depends on the order
the games happened to be played in. Bradley-Terry fits one strength per model
to all games at once by maximum likelihood, so order is irrelevant, and
bootstrapping the games gives honest confidence intervals. This is the same
move LMSYS made for Chatbot Arena. Both are reported on the familiar Elo
scale: a 400-point gap means 10:1 odds.
"""
import numpy as np

from src.elo import expected_score

SCALE = 400.0
BASE = 1000.0


def online_elo(games, score, n_models, k=4.0, start=BASE):
    """Sequential Elo over games in the given order."""
    r = np.full(n_models, start, float)
    for (a, b, _), s in zip(games, score):
        e = expected_score(r[a], r[b])
        r[a] += k * (s - e)
        r[b] -= k * (s - e)
    return r


def order_spread(games, score, n_models, n_orders=50, k=4.0, seed=7):
    """Std of each model's online Elo across random reorderings of the same games."""
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_orders):
        idx = rng.permutation(len(score))
        runs.append(online_elo(games[idx], score[idx], n_models, k=k))
    return np.std(runs, axis=0)


def pair_counts(games, score, n_models):
    """W[i, j] = points i scored against j (ties count half); N[i, j] = games between them."""
    a, b = games[:, 0], games[:, 1]
    W = np.zeros((n_models, n_models))
    N = np.zeros((n_models, n_models))
    np.add.at(W, (a, b), score)
    np.add.at(W, (b, a), 1.0 - score)
    np.add.at(N, (a, b), 1.0)
    np.add.at(N, (b, a), 1.0)
    return W, N


def bradley_terry(W, N, iters=1000, tol=1e-10):
    """Maximum-likelihood Bradley-Terry strengths via Hunter's MM algorithm.

    Returns ratings on the Elo scale, centred so the mean rating is BASE.
    """
    n = W.shape[0]
    p = np.ones(n)
    wins = W.sum(axis=1)
    for _ in range(iters):
        denom = (N / (p[:, None] + p[None, :])).sum(axis=1)
        new = wins / denom
        new /= np.exp(np.log(new).mean())
        if np.max(np.abs(new - p)) < tol:
            p = new
            break
        p = new
    r = SCALE * np.log10(p)
    return r - r.mean() + BASE


def win_probability(ratings):
    """P[i, j] = probability model i beats model j under the fitted ratings."""
    d = ratings[None, :] - ratings[:, None]
    return 1.0 / (1.0 + 10.0 ** (d / SCALE))


def bootstrap(games, score, n_models, n_boot=500, seed=7):
    """Bradley-Terry refit on games resampled with replacement.

    Returns (ratings[n_boot, n_models], ranks[n_boot, n_models]) where rank 1 is best.
    """
    rng = np.random.default_rng(seed)
    n = len(score)
    out = np.empty((n_boot, n_models))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        W, N = pair_counts(games[idx], score[idx], n_models)
        out[i] = bradley_terry(W, N)
    ranks = (-out).argsort(axis=1).argsort(axis=1) + 1
    return out, ranks
