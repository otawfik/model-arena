import pytest

from src import data, models


@pytest.fixture()
def small_data():
    texts, labels = data.generate(n_per_class=20, seed=3)
    return data.train_test_split(texts, labels, test_size=0.25, seed=4)


def test_all_models_train_and_predict(small_data):
    x_tr, x_te, y_tr, y_te = small_data
    for name in models.names():
        m = models.train(name, x_tr, y_tr)
        preds = models.predict(m, x_te)
        assert len(preds) == len(x_te)
        assert set(preds) <= set(data.CLASSES)


def test_build_unknown_raises():
    with pytest.raises(KeyError):
        models.build("not-a-model")


def test_logreg_beats_dummy(small_data):
    from sklearn.metrics import accuracy_score
    x_tr, x_te, y_tr, y_te = small_data
    acc_lr = accuracy_score(y_te, models.predict(models.train("logreg", x_tr, y_tr), x_te))
    acc_du = accuracy_score(y_te, models.predict(models.train("dummy", x_tr, y_tr), x_te))
    assert acc_lr > acc_du


def test_display_names_cover_registry():
    for name in models.names():
        assert name in models.DISPLAY_NAMES
