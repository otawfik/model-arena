"""Pairwise battle engine.

A battle pits two fitted models against each other on a held-out test set.
Each test sample is a point: a model wins the point when it alone predicts
the true label. The battle winner is the model with more points; equal
points is a tie. run_tournament plays every pair and keeps ELO ratings.
"""
import numpy as np

from src.elo import update_ratings


def battle(name_a, model_a, name_b, model_b, x_test, y_test):
    """One head-to-head matchup. Returns a record dict."""
    pred_a = list(model_a.predict(x_test))
    pred_b = list(model_b.predict(x_test))
    wins_a = wins_b = ties = 0
    for pa, pb, y in zip(pred_a, pred_b, y_test):
        a_ok, b_ok = (pa == y), (pb == y)
        if a_ok and not b_ok:
            wins_a += 1
        elif b_ok and not a_ok:
            wins_b += 1
        else:
            ties += 1
    if wins_a > wins_b:
        winner, score_a = name_a, 1.0
    elif wins_b > wins_a:
        winner, score_a = name_b, 0.0
    else:
        winner, score_a = "tie", 0.5
    return {
        "model_a": name_a,
        "model_b": name_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "winner": winner,
        "score_a": score_a,
        "n_samples": len(y_test),
    }


def run_tournament(models, x_test, y_test, rounds=3, k=32, seed=7,
                   start_rating=1200.0):
    """Round-robin over every model pair, `rounds` times, with ELO updates.

    models: dict of {name: fitted_model}. Returns
    {"ratings": {name: elo}, "records": [battle records]}.
    """
    names = list(models)
    ratings = {n: start_rating for n in names}
    records = []
    rng = np.random.RandomState(seed)
    for _ in range(rounds):
        order = rng.permutation(len(names))
        ordered = [names[i] for i in order]
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                rec = battle(a, models[a], b, models[b], x_test, y_test)
                ratings[a], ratings[b] = update_ratings(
                    ratings[a], ratings[b], rec["score_a"], k=k)
                records.append(rec)
    return {"ratings": ratings, "records": records}
