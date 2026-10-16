"""Run a full tournament and assemble the leaderboard.

Predict once per model, sample per-document games, then rate them three ways:
held-out accuracy (with a Wilson interval), online Elo (with its spread across
game orders, to show why it isn't the headline), and Bradley-Terry (with
bootstrap confidence intervals on both the rating and the rank). McNemar tests
say which accuracy gaps are real.
"""
from itertools import combinations

import numpy as np

from src import arena, ratings, stats
from src.models import DISPLAY_NAMES


def run(models, x_test, y_test, n_games=30000, n_boot=500, n_orders=50, k=4.0, seed=7):
    names, correct, _ = arena.correctness(models, x_test, y_test)
    m = len(names)
    games, score = arena.sample_games(correct, n_games, seed=seed)

    W, N = ratings.pair_counts(games, score, m)
    bt = ratings.bradley_terry(W, N)
    boot, boot_ranks = ratings.bootstrap(games, score, m, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    elo = ratings.online_elo(games, score, m, k=k)
    spread = ratings.order_spread(games, score, m, n_orders=n_orders, k=k, seed=seed)
    w, l, t = arena.record(games, score, m)

    n_docs = correct.shape[0]
    rows = []
    for i, name in enumerate(names):
        k_right = int(correct[:, i].sum())
        acc_lo, acc_hi = stats.wilson(k_right, n_docs)
        rows.append({
            "model": name,
            "display_name": DISPLAY_NAMES.get(name, name),
            "bt": round(float(bt[i]), 1),
            "ci_low": round(float(lo[i]), 1),
            "ci_high": round(float(hi[i]), 1),
            "rank_low": int(boot_ranks[:, i].min()),
            "rank_high": int(boot_ranks[:, i].max()),
            "elo_online": round(float(elo[i]), 1),
            "elo_order_std": round(float(spread[i]), 1),
            "accuracy": round(k_right / n_docs, 4),
            "acc_low": round(acc_lo, 4),
            "acc_high": round(acc_hi, 4),
            "wins": int(w[i]), "losses": int(l[i]), "ties": int(t[i]),
        })
    order = sorted(range(m), key=lambda i: -bt[i])
    rows = [rows[i] for i in order]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    pairs = list(combinations(range(m), 2))
    tests = [stats.mcnemar(correct[:, a], correct[:, b]) for a, b in pairs]
    adjusted = stats.holm([p for _, _, p in tests])
    mcnemar = [{
        "a": names[a], "b": names[b], "only_a": oa, "only_b": ob,
        "p": float(f"{p:.3g}"), "p_holm": float(f"{ph:.3g}"), "significant": bool(ph < 0.05),
    } for (a, b), (oa, ob, p), ph in zip(pairs, tests, adjusted)]

    prob = ratings.win_probability(bt)
    return {
        "meta": {"n_test": int(n_docs), "n_games": int(n_games), "n_boot": int(n_boot),
                 "n_orders": int(n_orders), "elo_k": k, "seed": seed},
        "rows": rows,
        "order": [names[i] for i in order],
        "win_prob": [[round(float(prob[i, j]), 3) for j in order] for i in order],
        "mcnemar": mcnemar,
    }
