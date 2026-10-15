import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, evaluate, models


def test_cross_validate_keys_and_ranges():
    texts, labels = data.generate(n_per_class=30, seed=6)
    res = evaluate.cross_validate("naive-bayes", texts, labels, cv=3, seed=6)
    assert res["model"] == "naive-bayes"
    assert 0.0 <= res["accuracy_mean"] <= 1.0
    assert 0.0 <= res["f1_macro_mean"] <= 1.0
    assert res["fit_time_mean_s"] >= 0.0


def test_cross_validate_beats_chance():
    texts, labels = data.generate(n_per_class=40, seed=6)
    res = evaluate.cross_validate("logreg", texts, labels, cv=3, seed=6)
    assert res["accuracy_mean"] > 0.5


def test_heldout_accuracy():
    texts, labels = data.generate(n_per_class=30, seed=8)
    x_tr, x_te, y_tr, y_te = data.train_test_split(texts, labels, seed=8)
    m = models.train("linear-svm", x_tr, y_tr)
    acc = evaluate.heldout_accuracy(m, x_te, y_te)
    assert 0.0 <= acc <= 1.0
    assert acc > 0.5
