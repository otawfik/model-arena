"""Flask demo: the arena leaderboard, real disagreements, and a live classifier.

The leaderboard and disagreement posts are precomputed by scripts/run_arena.py
and committed as JSON, so the page itself needs no models. /api/classify loads
the six fitted pipelines on first use: from ARENA_ARTIFACT or artifacts/ if
present, otherwise from the GitHub release (checksum-verified, cached in /tmp).
"""
import hashlib
import json
import os
import threading
import urllib.request

from flask import Flask, jsonify, render_template, request

from src.models import DISPLAY_NAMES, confidence

ROOT = os.path.dirname(os.path.abspath(__file__))
RELEASE_URL = "https://github.com/otawfik/model-arena/releases/download/v1.0/arena.joblib"
RELEASE_SHA256 = "d5f07cf65e3657259b0f9ef767d4f66605111859dd0568acb7fb74774fa7fbb3"
MAX_CHARS = 5000

app = Flask(__name__)


def _json(name):
    with open(os.path.join(ROOT, "models", name), encoding="utf-8") as f:
        return json.load(f)


BOARD = _json("leaderboard.json")
FIGHTS = _json("disagreements.json")

_arena = None
_lock = threading.Lock()


def _artifact_path():
    for candidate in (os.environ.get("ARENA_ARTIFACT"), os.path.join(ROOT, "artifacts", "arena.joblib")):
        if candidate and os.path.exists(candidate):
            return candidate
    cached = "/tmp/model-arena-v1.joblib"
    if not os.path.exists(cached):
        tmp = cached + ".part"
        urllib.request.urlretrieve(RELEASE_URL, tmp)
        with open(tmp, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != RELEASE_SHA256:
            os.remove(tmp)
            raise RuntimeError("model artifact checksum mismatch")
        os.replace(tmp, cached)
    return cached


def arena():
    """The fitted pipelines, loaded once per process."""
    global _arena
    with _lock:
        if _arena is None:
            import joblib
            _arena = joblib.load(_artifact_path())
    return _arena


@app.route("/")
def index():
    return render_template("index.html", board=BOARD, fights=FIGHTS, display=DISPLAY_NAMES)


@app.route("/api/leaderboard")
def api_leaderboard():
    return jsonify(BOARD)


@app.route("/api/classify", methods=["POST"])
def api_classify():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()[:MAX_CHARS]
    if not text:
        return jsonify({"error": "text is required"}), 400
    fitted = arena()["models"]
    out = []
    for name in BOARD["order"]:
        model = fitted[name]
        out.append({
            "model": name,
            "display_name": DISPLAY_NAMES.get(name, name),
            "label": str(model.predict([text])[0]),
            "confidence": confidence(model, [text])[0],
        })
    return jsonify({"predictions": out})


if __name__ == "__main__":
    app.run(debug=True)
