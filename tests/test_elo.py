from src import elo


def test_expected_score_equal_ratings():
    assert elo.expected_score(1200, 1200) == 0.5


def test_expected_score_stronger_favored():
    assert elo.expected_score(1600, 1200) > 0.9
    assert elo.expected_score(1200, 1600) < 0.1


def test_update_winner_gains_points():
    a, b = elo.update_ratings(1200, 1200, 1.0, k=32)
    assert a == 1216.0 and b == 1184.0


def test_update_upset_gains_more():
    upset_a, _ = elo.update_ratings(1000, 1400, 1.0, k=32)
    even_a, _ = elo.update_ratings(1200, 1200, 1.0, k=32)
    assert (upset_a - 1000) > (even_a - 1200)


def test_update_tie_moves_toward_equal():
    a, b = elo.update_ratings(1400, 1000, 0.5, k=32)
    assert a < 1400 < 1600
    assert b > 1000


def test_update_zero_sum():
    a, b = elo.update_ratings(1350, 1250, 1.0, k=32)
    assert round(a + b, 2) == 2600.0
