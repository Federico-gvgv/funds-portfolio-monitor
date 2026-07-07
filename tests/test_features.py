from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.features import build_features


def test_forward_return_target_construction() -> None:
    dates = pd.bdate_range("2020-01-01", periods=320)
    prices = pd.DataFrame({"SPY": np.arange(100.0, 420.0)}, index=dates)

    features = build_features(prices, forward_return_horizon=21)
    row = features.iloc[0]
    date = row["date"]
    expected = prices.loc[dates[dates.get_loc(date) + 21], "SPY"] / prices.loc[date, "SPY"] - 1.0

    assert row["forward_return_21"] == expected


def test_rolling_features_use_current_and_past_prices_only() -> None:
    dates = pd.bdate_range("2020-01-01", periods=320)
    base = pd.Series(100.0, index=dates)
    base.iloc[252] = 200.0
    future_changed = base.copy()
    future_changed.iloc[253:] = 1000.0

    original = build_features(pd.DataFrame({"SPY": base}), forward_return_horizon=21)
    changed = build_features(pd.DataFrame({"SPY": future_changed}), forward_return_horizon=21)
    date = dates[252]

    original_row = original[original["date"] == date].iloc[0]
    changed_row = changed[changed["date"] == date].iloc[0]

    assert original_row["momentum_21"] == changed_row["momentum_21"]
    assert original_row["trend_200"] == changed_row["trend_200"]
    assert original_row["drawdown_252"] == changed_row["drawdown_252"]
    assert original_row["forward_return_21"] != changed_row["forward_return_21"]
