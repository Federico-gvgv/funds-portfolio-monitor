from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.backtest import run_model_strategy
from quant_portfolio.features import build_features


def test_synthetic_backtest_runs_end_to_end() -> None:
    prices = _synthetic_prices()
    features = build_features(prices, forward_return_horizon=21)
    result = run_model_strategy(
        "Top-k Momentum",
        "momentum",
        prices,
        features,
        _config(transaction_cost_bps=10),
        random_seed=42,
    )

    assert not result.returns.empty
    assert not result.equity.empty
    assert not result.weights.empty
    assert result.weights.sum(axis=1).round(10).eq(1.0).all()


def test_transaction_costs_reduce_performance_when_turnover_is_positive() -> None:
    prices = _synthetic_prices()
    features = build_features(prices, forward_return_horizon=21)
    no_cost = run_model_strategy(
        "Top-k Momentum",
        "momentum",
        prices,
        features,
        _config(transaction_cost_bps=0),
        random_seed=42,
    )
    with_cost = run_model_strategy(
        "Top-k Momentum",
        "momentum",
        prices,
        features,
        _config(transaction_cost_bps=50),
        random_seed=42,
    )

    assert with_cost.turnover.sum() > 0.0
    assert with_cost.equity.iloc[-1] < no_cost.equity.iloc[-1]


def _synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=900)
    t = np.arange(len(dates))
    prices = pd.DataFrame(
        {
            "SPY": 100.0 * np.exp(0.0003 * t + 0.02 * np.sin(t / 25)),
            "QQQ": 100.0 * np.exp(0.0004 * t + 0.03 * np.sin(t / 30 + 1.5)),
            "TLT": 100.0 * np.exp(0.0001 * t + 0.04 * np.sin(t / 40 + 3.0)),
            "GLD": 100.0 * np.exp(0.0002 * t + 0.035 * np.sin(t / 35 + 4.0)),
        },
        index=dates,
    )
    return prices


def _config(transaction_cost_bps: int) -> dict:
    return {
        "rebalance_frequency": "M",
        "top_k": 2,
        "transaction_cost_bps": transaction_cost_bps,
        "initial_capital": 1.0,
        "train_start": "2018-01-01",
        "test_start": "2019-01-01",
        "min_train_months": 12,
    }
