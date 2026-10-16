import os
import sys

import joblib
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import data, models  # noqa: E402


@pytest.fixture(scope="session")
def small_data():
    texts, labels = data.generate(n_per_class=40, seed=11)
    return data.train_test_split(texts, labels, seed=12)


@pytest.fixture(scope="session")
def fitted(small_data):
    x_tr, _, y_tr, _ = small_data
    return {n: models.train(n, x_tr, y_tr) for n in models.names()}


@pytest.fixture(scope="session")
def client(tmp_path_factory, fitted):
    """Flask test client backed by a toy artifact, so tests never hit the network."""
    path = tmp_path_factory.mktemp("art") / "arena.joblib"
    joblib.dump({"models": fitted, "labels": data.CLASSES}, path)
    os.environ["ARENA_ARTIFACT"] = str(path)
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()
