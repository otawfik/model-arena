import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module


def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_index_loads():
    c = client()
    resp = c.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Model Arena" in html
    assert "Leaderboard" in html


def test_index_shows_all_models():
    c = client()
    html = c.get("/").get_data(as_text=True)
    for name in ("Logistic Regression", "Naive Bayes", "Dummy"):
        assert name in html


def test_battle_strong_wins():
    c = client()
    resp = c.post("/battle", data={
        "text": "the striker scored the winning goal in the final match",
        "expected": "sports",
        "model_a": "logreg",
        "model_b": "dummy",
    })
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "sports" in html
    assert "wins the round" in html


def test_battle_tie_when_agree():
    c = client()
    resp = c.post("/battle", data={
        "text": "deploy the neural model to the cloud server",
        "expected": "tech",
        "model_a": "logreg",
        "model_b": "logreg",
    })
    assert resp.status_code == 200
    assert "Draw." in resp.get_data(as_text=True)


def test_battle_unknown_model_400():
    c = client()
    resp = c.post("/battle", data={
        "text": "hello", "expected": "sports",
        "model_a": "nope", "model_b": "logreg",
    })
    assert resp.status_code == 400


def test_api_leaderboard_json():
    c = client()
    resp = c.get("/api/leaderboard")
    assert resp.status_code == 200
    rows = resp.get_json()
    assert isinstance(rows, list) and len(rows) == 6
    elos = [r["elo"] for r in rows]
    assert elos == sorted(elos, reverse=True)
