from __future__ import annotations

import pandas as pd

from quant_portfolio.portfolio import compute_turnover, construct_weights, select_top_k


def test_top_k_selection_and_weights_sum_to_one() -> None:
    scores = pd.Series({"SPY": 0.2, "QQQ": 0.4, "TLT": -0.1, "GLD": 0.3})

    assert select_top_k(scores, top_k=2) == ["QQQ", "GLD"]

    weights = construct_weights(scores, top_k=2)
    assert weights.loc["QQQ"] == 0.5
    assert weights.loc["GLD"] == 0.5
    assert weights.loc["SPY"] == 0.0
    assert weights.sum() == 1.0


def test_turnover_calculation() -> None:
    previous = pd.Series({"SPY": 0.5, "QQQ": 0.5})
    new = pd.Series({"SPY": 0.0, "QQQ": 0.5, "TLT": 0.5})

    assert compute_turnover(previous, new) == 0.5
