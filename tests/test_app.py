def test_index_renders_leaderboard(client):
    html = client.get("/").get_data(as_text=True)
    assert "Model Arena" in html and "Leaderboard" in html
    for name in ("Complement Naive Bayes", "Logistic Regression", "Dummy"):
        assert name in html


def test_api_leaderboard(client):
    board = client.get("/api/leaderboard").get_json()
    assert len(board["rows"]) == 6
    bts = [r["bt"] for r in board["rows"]]
    assert bts == sorted(bts, reverse=True)


def test_classify_returns_all_six(client):
    r = client.post("/api/classify", json={"text": "the coach praised the striker after the goal"})
    assert r.status_code == 200
    preds = r.get_json()["predictions"]
    assert len(preds) == 6
    svm = next(p for p in preds if p["model"] == "linear-svm")
    assert svm["confidence"] is None
    lr = next(p for p in preds if p["model"] == "logreg")
    assert lr["label"] == "sports" and 0 < lr["confidence"] <= 1


def test_classify_requires_text(client):
    assert client.post("/api/classify", json={"text": "  "}).status_code == 400
    assert client.post("/api/classify", data="nope").status_code == 400
