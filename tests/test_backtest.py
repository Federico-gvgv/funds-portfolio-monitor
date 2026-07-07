from __future__ import annotations

import os
import numpy as np
import pandas as pd
import subprocess
import sys

from pathlib import Path

from quant_portfolio.backtest import (
    align_strategy_returns,
    equity_curves,
    results_to_metrics,
    run_model_strategy,
    run_static_strategy,
)
from quant_portfolio.features import build_features


ROOT = Path(__file__).resolve().parents[1]


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


def test_training_window_lags_labels_by_forward_horizon(monkeypatch) -> None:
    prices = _synthetic_prices()
    horizon = 21
    features = build_features(prices, forward_return_horizon=horizon)
    latest_training_dates: list[pd.Timestamp] = []

    class SpyModel:
        def fit(self, features_train: pd.DataFrame) -> "SpyModel":
            latest_training_dates.append(features_train["date"].max())
            return self

        def predict_scores(self, features_current: pd.DataFrame) -> pd.Series:
            return pd.Series(np.arange(len(features_current)), index=features_current["ticker"], name="score")

    monkeypatch.setattr("quant_portfolio.backtest.make_model", lambda *args, **kwargs: SpyModel())

    result = run_model_strategy(
        "Spy",
        "ridge",
        prices,
        features,
        _config(transaction_cost_bps=0, forward_return_horizon=horizon),
        random_seed=42,
    )

    assert latest_training_dates
    assert len(latest_training_dates) == len(result.weights.index)
    for latest_training_date, rebalance_date in zip(latest_training_dates, result.weights.index):
        assert latest_training_date <= rebalance_date - pd.tseries.offsets.BDay(horizon)


def test_metrics_and_equity_use_common_aligned_date_range() -> None:
    prices = _synthetic_prices()
    features = build_features(prices, forward_return_horizon=21)
    config = _config(transaction_cost_bps=10)
    results = {
        "SPY Buy & Hold": run_static_strategy("SPY Buy & Hold", prices, {"SPY": 1.0}, config),
        "Top-k Momentum": run_model_strategy(
            "Top-k Momentum",
            "momentum",
            prices,
            features,
            config,
            random_seed=42,
        ),
    }

    aligned_returns = align_strategy_returns(results)
    metrics = results_to_metrics(results)
    equity = equity_curves(results)

    assert not metrics.empty
    assert aligned_returns.index.min() == equity.index.min()
    assert aligned_returns.index.max() == equity.index.max()
    assert aligned_returns.notna().all().all()
    assert equity.notna().all().all()
    assert all(result.returns.index.min() <= aligned_returns.index.min() for result in results.values())
    assert all(result.returns.index.max() >= aligned_returns.index.max() for result in results.values())


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


def test_static_turnover_reports_initial_allocation_separately() -> None:
    prices = _synthetic_prices()
    no_cost = run_static_strategy("SPY Buy & Hold", prices, {"SPY": 1.0}, _config(transaction_cost_bps=0))
    with_cost = run_static_strategy("SPY Buy & Hold", prices, {"SPY": 1.0}, _config(transaction_cost_bps=50))
    metrics = results_to_metrics({"SPY Buy & Hold": with_cost})

    assert with_cost.initial_turnover > 0.0
    assert metrics.loc["SPY Buy & Hold", "initial_turnover"] == with_cost.initial_turnover
    assert metrics.loc["SPY Buy & Hold", "average_turnover"] == 0.0
    assert with_cost.equity.iloc[-1] < no_cost.equity.iloc[-1]


def test_backtest_and_report_scripts_run_on_bundled_data() -> None:
    env = os.environ.copy()
    env["QUANT_PORTFOLIO_SKIP_REFRESH_DOWNLOAD"] = "1"

    subprocess.run(
        [sys.executable, "scripts/run_backtest.py", "--config", "configs/backtest.yaml"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "scripts/generate_report.py"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


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


def _config(transaction_cost_bps: int, forward_return_horizon: int = 21) -> dict:
    return {
        "rebalance_frequency": "M",
        "top_k": 2,
        "transaction_cost_bps": transaction_cost_bps,
        "initial_capital": 1.0,
        "train_start": "2018-01-01",
        "test_start": "2019-01-01",
        "min_train_months": 12,
        "forward_return_horizon": forward_return_horizon,
    }
