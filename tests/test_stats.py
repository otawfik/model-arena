import numpy as np
import pytest

from src import stats


def test_mcnemar_counts_discordant_pairs():
    a = np.array([1, 1, 1, 0, 0, 1], bool)
    b = np.array([1, 0, 0, 1, 0, 1], bool)
    only_a, only_b, p = stats.mcnemar(a, b)
    assert (only_a, only_b) == (2, 1)
    assert p == pytest.approx(1.0)


def test_mcnemar_detects_a_clear_gap():
    a = np.array([True] * 60 + [False] * 5)
    b = np.array([False] * 60 + [True] * 5)
    assert stats.mcnemar(a, b)[2] < 1e-8


def test_mcnemar_identical_models():
    a = np.array([True, False, True])
    assert stats.mcnemar(a, a) == (0, 0, 1.0)


def test_holm_matches_hand_computation():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_monotone_and_capped():
    adj = stats.holm([0.5, 0.001, 0.2, 0.9])
    assert max(adj) <= 1.0
    assert adj[1] <= adj[2] <= adj[0] <= adj[3]


def test_wilson_interval():
    lo, hi = stats.wilson(50, 100)
    assert lo < 0.5 < hi
    assert hi - lo == pytest.approx(0.192, abs=0.005)
    assert stats.wilson(0, 0) == (0.0, 0.0)
