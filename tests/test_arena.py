import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import arena, data, models


def _two_fitted():
    texts, labels = data.generate(n_per_class=25, seed=11)
    x_tr, x_te, y_tr, y_te = data.train_test_split(texts, labels, seed=12)
    fitted = {
        "logreg": models.train("logreg", x_tr, y_tr),
        "dummy": models.train("dummy", x_tr, y_tr),
    }
    return fitted, x_te, y_te


def test_battle_record_fields():
    fitted, x_te, y_te = _two_fitted()
    rec = arena.battle("logreg", fitted["logreg"], "dummy", fitted["dummy"], x_te, y_te)
    assert rec["wins_a"] + rec["wins_b"] + rec["ties"] == len(y_te)
    assert rec["winner"] in ("logreg", "dummy", "tie")
    assert rec["n_samples"] == len(y_te)


def test_battle_strong_beats_weak():
    fitted, x_te, y_te = _two_fitted()
    rec = arena.battle("logreg", fitted["logreg"], "dummy", fitted["dummy"], x_te, y_te)
    assert rec["winner"] == "logreg"
    assert rec["score_a"] == 1.0


def test_tournament_elo_moves_and_ranks():
    texts, labels = data.generate(n_per_class=25, seed=11)
    x_tr, x_te2, y_tr, y_te2 = data.train_test_split(texts, labels, seed=12)
    fitted = {
        "logreg": models.train("logreg", x_tr, y_tr),
        "naive-bayes": models.train("naive-bayes", x_tr, y_tr),
        "dummy": models.train("dummy", x_tr, y_tr),
    }
    res = arena.run_tournament(fitted, x_te2, y_te2, rounds=2, seed=9)
    assert len(res["records"]) == 2 * 3  # rounds * C(3,2)
    assert res["ratings"]["logreg"] > 1200
    assert res["ratings"]["dummy"] < 1200


def test_tournament_deterministic():
    texts, labels = data.generate(n_per_class=25, seed=11)
    x_tr, x_te, y_tr, y_te = data.train_test_split(texts, labels, seed=12)
    fitted = {n: models.train(n, x_tr, y_tr) for n in ("logreg", "knn")}
    r1 = arena.run_tournament(fitted, x_te, y_te, seed=9)
    r2 = arena.run_tournament(fitted, x_te, y_te, seed=9)
    assert r1["ratings"] == r2["ratings"]
