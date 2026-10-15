"""Evaluation harness: stratified cross-validation metrics for a model."""
import time

from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.models import build


def cross_validate(name, x, y, cv=5, seed=7):
    """5-fold (default) stratified CV: accuracy, macro F1, mean fit time."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    accs, f1s, times = [], [], []
    for tr_idx, va_idx in skf.split(x, y):
        model = build(name)
        t0 = time.perf_counter()
        model.fit([x[i] for i in tr_idx], [y[i] for i in tr_idx])
        times.append(time.perf_counter() - t0)
        pred = model.predict([x[i] for i in va_idx])
        y_va = [y[i] for i in va_idx]
        accs.append(accuracy_score(y_va, pred))
        f1s.append(f1_score(y_va, pred, average="macro", zero_division=0))
    return {
        "model": name,
        "accuracy_mean": round(sum(accs) / len(accs), 4),
        "accuracy_std": round(_std(accs), 4),
        "f1_macro_mean": round(sum(f1s) / len(f1s), 4),
        "fit_time_mean_s": round(sum(times) / len(times), 3),
    }


def heldout_accuracy(model, x_test, y_test):
    """Accuracy of an already-fitted model on a held-out set."""
    pred = model.predict(x_test)
    return round(float(accuracy_score(y_test, pred)), 4)


def _std(vals):
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
