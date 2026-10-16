from src import data


def test_generate_sizes_and_labels():
    texts, labels = data.generate(n_per_class=20, seed=1)
    assert len(texts) == 100
    assert set(labels) == set(data.CLASSES)
    for cls in data.CLASSES:
        assert labels.count(cls) == 20


def test_generate_deterministic():
    t1, l1 = data.generate(n_per_class=10, seed=5)
    t2, l2 = data.generate(n_per_class=10, seed=5)
    assert t1 == t2 and l1 == l2


def test_generate_shuffled_not_grouped():
    _, labels = data.generate(n_per_class=20, seed=1)
    assert labels[:5] != ["sports"] * 5


def test_train_test_split_sizes():
    texts, labels = data.generate(n_per_class=40, seed=1)
    x_tr, x_te, y_tr, y_te = data.train_test_split(texts, labels, test_size=0.25, seed=2)
    assert len(x_tr) == 150 and len(x_te) == 50


def test_train_test_split_class_balance():
    texts, labels = data.generate(n_per_class=40, seed=1)
    _, _, _, y_te = data.train_test_split(texts, labels, test_size=0.25, seed=2)
    for cls in data.CLASSES:
        assert y_te.count(cls) == 10
