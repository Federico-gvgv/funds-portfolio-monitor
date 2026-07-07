from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_portfolio.metrics import summarize_returns
from quant_portfolio.models import make_model
from quant_portfolio.portfolio import compute_turnover, construct_weights


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series


def run_all_strategies(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    config: dict,
) -> dict[str, BacktestResult]:
    backtest_config = config["backtest"]
    random_seed = config.get("models", {}).get("random_seed", 42)
    tickers = list(prices.columns)
    results = {
        "SPY Buy & Hold": run_static_strategy(
            "SPY Buy & Hold",
            prices,
            {"SPY": 1.0},
            backtest_config,
        ),
        "Equal Weight Universe": run_equal_weight_strategy(
            prices,
            backtest_config,
        ),
        "Top-k Momentum": run_model_strategy(
            "Top-k Momentum",
            "momentum",
            prices,
            features,
            backtest_config,
            random_seed,
            tickers,
        ),
        "Top-k Vol-Adjusted Momentum": run_model_strategy(
            "Top-k Vol-Adjusted Momentum",
            "vol_adjusted_momentum",
            prices,
            features,
            backtest_config,
            random_seed,
            tickers,
        ),
        "Ridge Predicted Return": run_model_strategy(
            "Ridge Predicted Return",
            "ridge",
            prices,
            features,
            backtest_config,
            random_seed,
            tickers,
        ),
        "Gradient Boosting Predicted Return": run_model_strategy(
            "Gradient Boosting Predicted Return",
            "gradient_boosting",
            prices,
            features,
            backtest_config,
            random_seed,
            tickers,
        ),
    }
    return results


def run_model_strategy(
    name: str,
    model_name: str,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    backtest_config: dict,
    random_seed: int,
    tickers: list[str] | None = None,
) -> BacktestResult:
    returns = prices.pct_change().fillna(0.0)
    tickers = tickers or list(prices.columns)
    rebalance_dates = _rebalance_dates(features["date"], backtest_config["rebalance_frequency"])
    first_date = _first_eligible_date(
        prices.index.min(),
        backtest_config["test_start"],
        backtest_config["min_train_months"],
    )
    rebalance_dates = [date for date in rebalance_dates if date >= first_date]

    previous_weights = pd.Series(0.0, index=tickers, dtype=float)
    daily_returns: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    turnover_values: dict[pd.Timestamp, float] = {}

    for index, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = rebalance_dates[index + 1] if index + 1 < len(rebalance_dates) else returns.index.max()
        train = features[
            (features["date"] < rebalance_date)
            & (features["date"] >= pd.Timestamp(backtest_config["train_start"]))
        ]
        current = features[features["date"] == rebalance_date]
        if train.empty or current.empty:
            continue

        model = make_model(model_name, random_seed=random_seed)
        model.fit(train)
        scores = model.predict_scores(current).reindex(tickers)
        volatility = current.set_index("ticker").get("volatility_60")
        weights = construct_weights(
            scores=scores,
            top_k=backtest_config["top_k"],
            volatility=volatility,
            weighting="equal",
        ).reindex(tickers, fill_value=0.0)
        turnover = compute_turnover(previous_weights, weights)
        period_returns = _period_returns(returns, rebalance_date, next_rebalance, weights, backtest_config, turnover)
        if period_returns.empty:
            continue

        daily_returns.append(period_returns)
        weight_row = weights.copy()
        weight_row.name = rebalance_date
        weight_rows.append(weight_row)
        turnover_values[rebalance_date] = turnover
        previous_weights = weights

    return _build_result(name, daily_returns, weight_rows, turnover_values)


def run_static_strategy(
    name: str,
    prices: pd.DataFrame,
    allocations: dict[str, float],
    backtest_config: dict,
) -> BacktestResult:
    returns = prices.pct_change().fillna(0.0)
    first_date = _first_eligible_date(
        prices.index.min(),
        backtest_config["test_start"],
        backtest_config["min_train_months"],
    )
    weights = pd.Series(0.0, index=prices.columns, dtype=float)
    for ticker, weight in allocations.items():
        if ticker in weights.index:
            weights.loc[ticker] = weight
    strategy_returns = returns.loc[returns.index >= first_date].mul(weights, axis=1).sum(axis=1)
    weights_frame = pd.DataFrame([weights], index=[strategy_returns.index.min()]) if not strategy_returns.empty else pd.DataFrame()
    turnover = pd.Series([compute_turnover(pd.Series(0.0, index=prices.columns), weights)], index=weights_frame.index)
    return BacktestResult(
        name=name,
        returns=strategy_returns.rename(name),
        equity=(1.0 + strategy_returns).cumprod().rename(name),
        weights=weights_frame,
        turnover=turnover.rename(name),
    )


def run_equal_weight_strategy(prices: pd.DataFrame, backtest_config: dict) -> BacktestResult:
    weight = 1.0 / len(prices.columns)
    return run_static_strategy(
        "Equal Weight Universe",
        prices,
        {ticker: weight for ticker in prices.columns},
        backtest_config,
    )


def results_to_metrics(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        row = {"strategy": name}
        row.update(summarize_returns(result.returns, result.turnover))
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


def equity_curves(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.concat({name: result.equity for name, result in results.items()}, axis=1)


def weights_to_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    frames = []
    for name, result in results.items():
        if result.weights.empty:
            continue
        frame = result.weights.copy()
        frame.index.name = "date"
        frame = frame.reset_index().melt(id_vars="date", var_name="ticker", value_name="weight")
        frame.insert(0, "strategy", name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _period_returns(
    returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    next_rebalance: pd.Timestamp,
    weights: pd.Series,
    backtest_config: dict,
    turnover: float,
) -> pd.Series:
    period = returns[(returns.index > rebalance_date) & (returns.index <= next_rebalance)]
    if period.empty:
        return pd.Series(dtype=float)
    strategy_returns = period.mul(weights, axis=1).sum(axis=1)
    cost = backtest_config["transaction_cost_bps"] / 10000.0 * turnover
    strategy_returns.iloc[0] -= cost
    return strategy_returns


def _build_result(
    name: str,
    daily_returns: list[pd.Series],
    weight_rows: list[pd.Series],
    turnover_values: dict[pd.Timestamp, float],
) -> BacktestResult:
    returns = pd.concat(daily_returns).sort_index() if daily_returns else pd.Series(dtype=float)
    returns = returns[~returns.index.duplicated(keep="first")].rename(name)
    weights = pd.DataFrame(weight_rows).sort_index() if weight_rows else pd.DataFrame()
    turnover = pd.Series(turnover_values, name=name).sort_index()
    equity = (1.0 + returns).cumprod().rename(name)
    return BacktestResult(name=name, returns=returns, equity=equity, weights=weights, turnover=turnover)


def _rebalance_dates(dates: pd.Series, frequency: str) -> list[pd.Timestamp]:
    unique_dates = pd.Series(pd.to_datetime(dates).sort_values().unique())
    if unique_dates.empty:
        return []
    frequency = "ME" if frequency == "M" else frequency
    frame = pd.DataFrame(index=pd.DatetimeIndex(unique_dates))
    return list(frame.resample(frequency).last().dropna().index)


def _first_eligible_date(
    data_start: pd.Timestamp,
    test_start: str,
    min_train_months: int,
) -> pd.Timestamp:
    min_train_date = pd.Timestamp(data_start) + pd.DateOffset(months=min_train_months)
    return max(pd.Timestamp(test_start), min_train_date)
