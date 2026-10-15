"""Flask demo: arena leaderboard + interactive head-to-head battles."""
import json
import os

from flask import Flask, jsonify, render_template, request

from src import data, leaderboard as lb, models
from src.arena import battle

app = Flask(__name__)

MODELS = {}
BOARD = []

SAMPLE_TEXTS = [
    ("the coach praised the striker after the winning goal", "sports"),
    ("deployed the neural model to the cloud api endpoint", "tech"),
    ("simmer the garlic sauce then bake until crispy", "food"),
    ("boarding the flight with my passport and luggage", "travel"),
    ("the stock dividend boosted the portfolio revenue", "finance"),
]


def _load():
    global BOARD
    texts, labels = data.generate(n_per_class=120, seed=42)
    x_train, x_test, y_train, y_test = data.train_test_split(texts, labels)
    for name in models.names():
        MODELS[name] = models.train(name, x_train, y_train)
    board_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "leaderboard.json")
    if os.path.exists(board_path):
        with open(board_path, encoding="utf-8") as f:
            BOARD = json.load(f)
    else:
        from src import arena as arena_mod, evaluate
        metrics = {n: evaluate.cross_validate(n, x_train, y_train)
                   for n in models.names()}
        res = arena_mod.run_tournament(MODELS, x_test, y_test)
        BOARD = lb.build(metrics, res["ratings"], res["records"])


_load()


@app.route("/")
def index():
    return render_template(
        "index.html",
        board=BOARD,
        names=models.names(),
        display=models.DISPLAY_NAMES,
        classes=data.CLASSES,
        samples=SAMPLE_TEXTS,
    )


@app.route("/battle", methods=["POST"])
def do_battle():
    text = request.form.get("text", "").strip()
    expected = request.form.get("expected", "")
    name_a = request.form.get("model_a", "logreg")
    name_b = request.form.get("model_b", "naive-bayes")
    if name_a not in MODELS or name_b not in MODELS:
        return "unknown model", 400
    pred_a = models.predict(MODELS[name_a], [text])[0]
    pred_b = models.predict(MODELS[name_b], [text])[0]
    ok_a = pred_a == expected
    ok_b = pred_b == expected
    if ok_a and not ok_b:
        winner = name_a
    elif ok_b and not ok_a:
        winner = name_b
    else:
        winner = "tie"
    return render_template(
        "index.html",
        board=BOARD,
        names=models.names(),
        display=models.DISPLAY_NAMES,
        classes=data.CLASSES,
        samples=SAMPLE_TEXTS,
        result={
            "text": text, "expected": expected,
            "name_a": name_a, "name_b": name_b,
            "pred_a": pred_a, "pred_b": pred_b,
            "ok_a": ok_a, "ok_b": ok_b, "winner": winner,
        },
    )


@app.route("/api/leaderboard")
def api_leaderboard():
    return jsonify(BOARD)


if __name__ == "__main__":
    app.run(debug=True)
