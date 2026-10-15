"""Leaderboard builder: merge CV metrics, ELO ratings, and battle records."""
import json

from src.models import DISPLAY_NAMES


def build(metrics, ratings, records):
    """Ranked rows (ELO desc) combining everything the arena knows."""
    rows = []
    for name, m in metrics.items():
        wins = sum(1 for r in records if r["winner"] == name)
        losses = sum(
            1 for r in records
            if r["winner"] not in (name, "tie")
            and name in (r["model_a"], r["model_b"]))
        ties = sum(
            1 for r in records
            if r["winner"] == "tie"
            and name in (r["model_a"], r["model_b"]))
        rows.append({
            "model": name,
            "display_name": DISPLAY_NAMES.get(name, name),
            "elo": round(ratings.get(name, 0.0), 1),
            "cv_accuracy": m.get("accuracy_mean"),
            "cv_f1_macro": m.get("f1_macro_mean"),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        })
    rows.sort(key=lambda r: r["elo"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def save(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
