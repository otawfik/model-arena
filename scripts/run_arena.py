"""Run the full arena on 20 Newsgroups and save everything the demo needs.

    python scripts/run_arena.py            # ~1-2 minutes on a laptop

Writes:
  models/leaderboard.json     ratings, intervals, McNemar tests, win probabilities
  models/disagreements.json   held-out posts where the strongest models split
  artifacts/arena.joblib      the six fitted pipelines (published as a release asset)
"""
import json
import os
import re
import sys
import time

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import arena, data, models, tournament  # noqa: E402

N_DISAGREEMENTS = 150


EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


def snippet(text, limit=520):
    """Whitespace-collapsed, truncated, and with email addresses from old signatures removed."""
    text = EMAIL.sub("[email]", re.sub(r"\s+", " ", text)).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def main():
    x_train, x_test, y_train, y_test, labels = data.load_newsgroups()
    print(f"20 Newsgroups: {len(x_train)} train / {len(x_test)} test, {len(labels)} topics")

    fitted, fit_s = {}, {}
    for name in models.names():
        t0 = time.perf_counter()
        fitted[name] = models.train(name, x_train, y_train)
        fit_s[name] = round(time.perf_counter() - t0, 2)
        print(f"  trained {name:<14} {fit_s[name]:>6.2f}s")

    result = tournament.run(fitted, x_test, y_test)
    result["meta"].update({"dataset": "20 Newsgroups (by-date split, headers/footers/quotes removed)",
                           "n_train": len(x_train), "n_classes": len(labels)})
    for row in result["rows"]:
        row["fit_seconds"] = fit_s[row["model"]]

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    with open(os.path.join(ROOT, "models", "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # posts where the top three disagree, for the demo's "show me a fight" button
    names, correct, preds = arena.correctness(fitted, x_test, y_test)
    top = result["order"][:3]
    split = [i for i in range(len(x_test))
             if len({preds[n][i] for n in top}) > 1 and len(x_test[i].strip()) > 160]
    rng = np.random.default_rng(7)
    picks = sorted(rng.choice(split, size=min(N_DISAGREEMENTS, len(split)), replace=False))
    fights = [{"text": snippet(x_test[i]), "truth": y_test[i],
               "preds": {n: str(preds[n][i]) for n in names}} for i in picks]
    with open(os.path.join(ROOT, "models", "disagreements.json"), "w", encoding="utf-8") as f:
        json.dump(fights, f, indent=1)

    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    art = os.path.join(ROOT, "artifacts", "arena.joblib")
    joblib.dump({"models": fitted, "labels": labels}, art, compress=3)

    print(f"\n{'rank':<5}{'model':<26}{'BT':>7}  {'95% CI':<15}{'acc':>7}  {'online Elo ± order sd'}")
    for r in result["rows"]:
        print(f"{r['rank']:<5}{r['display_name']:<26}{r['bt']:>7.1f}  "
              f"[{r['ci_low']:.0f}, {r['ci_high']:.0f}]{'':<4}{r['accuracy']:>7.3f}  "
              f"{r['elo_online']:.0f} ± {r['elo_order_std']:.1f}")
    sig = sum(t["significant"] for t in result["mcnemar"])
    print(f"\n{sig}/{len(result['mcnemar'])} pairs differ significantly (McNemar, Holm-adjusted p < .05)")
    print(f"{len(split)} test posts split the top three; saved {len(fights)}")
    print(f"artifact: {os.path.getsize(art) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
