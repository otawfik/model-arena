from src import tournament


def test_tournament_result_is_complete_and_ranked(fitted, small_data):
    _, x_te, _, y_te = small_data
    res = tournament.run(fitted, x_te, y_te, n_games=3000, n_boot=40, n_orders=5)
    rows = res["rows"]
    assert [r["rank"] for r in rows] == list(range(1, len(fitted) + 1))
    assert [r["bt"] for r in rows] == sorted((r["bt"] for r in rows), reverse=True)
    for r in rows:
        assert r["ci_low"] <= r["bt"] <= r["ci_high"]
        assert r["acc_low"] <= r["accuracy"] <= r["acc_high"]
        assert 1 <= r["rank_low"] <= r["rank_high"] <= len(fitted)
    assert len(res["mcnemar"]) == len(fitted) * (len(fitted) - 1) // 2
    assert len(res["win_prob"]) == len(fitted)
    assert res["order"] == [r["model"] for r in rows]


def test_dummy_finishes_last(fitted, small_data):
    _, x_te, _, y_te = small_data
    res = tournament.run(fitted, x_te, y_te, n_games=3000, n_boot=20, n_orders=3)
    assert res["rows"][-1]["model"] == "dummy"
