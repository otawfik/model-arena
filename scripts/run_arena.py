"""Run the full arena tournament and save the leaderboard.

Generates the dataset, trains every contender, runs cross-validation,
plays round-robin battles with ELO updates, and writes
models/leaderboard.json + models/metrics.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import arena, data, evaluate, leaderboard, models  # noqa: E402

DATA_SEED = 42
SPLIT_SEED = 7
N_PER_CLASS = 200
ROUNDS = 3


def main():
    texts, labels = data.generate(n_per_class=N_PER_CLASS, seed=DATA_SEED)
    x_train, x_test, y_train, y_test = data.train_test_split(
        texts, labels, test_size=0.25, seed=SPLIT_SEED)
    print(f"dataset: {len(texts)} texts, {len(x_train)} train / {len(x_test)} test")

    fitted = {}
    metrics = {}
    for name in models.names():
        print(f"training {name} ...")
        fitted[name] = models.train(name, x_train, y_train)
        metrics[name] = evaluate.cross_validate(name, x_train, y_train)
        metrics[name]["heldout_accuracy"] = evaluate.heldout_accuracy(
            fitted[name], x_test, y_test)

    result = arena.run_tournament(fitted, x_test, y_test, rounds=ROUNDS)
    rows = leaderboard.build(metrics, result["ratings"], result["records"])

    out_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(out_dir, exist_ok=True)
    leaderboard.save(rows, os.path.join(out_dir, "leaderboard.json"))
    with open(os.path.join(out_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nLEADERBOARD (ELO)")
    print(f"{'rank':<5}{'model':<22}{'elo':<9}{'cv acc':<9}{'w-l-t'}")
    for r in rows:
        print(f"{r['rank']:<5}{r['display_name']:<22}{r['elo']:<9}"
              f"{r['cv_accuracy']:<9}{r['wins']}-{r['losses']}-{r['ties']}")
    print(f"\nsaved models/leaderboard.json and models/metrics.json "
          f"({len(result['records'])} battles)")


if __name__ == "__main__":
    main()
