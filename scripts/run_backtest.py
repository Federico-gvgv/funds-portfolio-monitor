from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_portfolio.backtest import (
    align_strategy_returns,
    equity_curves,
    results_to_metrics,
    run_all_strategies,
    weights_to_frame,
)
from quant_portfolio.config import load_config
from quant_portfolio.data import download_prices, load_legacy_ticker_csvs, load_prices, save_prices
from quant_portfolio.features import build_features
from quant_portfolio.plots import (
    plot_allocation_heatmap,
    plot_drawdowns,
    plot_equity_curves,
    plot_rolling_sharpe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward ETF allocation backtest.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    prices = _load_or_download_prices(config)
    prices = _validate_or_refresh_data_start(config, prices)
    prices = prices.loc[prices.index >= pd.Timestamp(config["data"]["start_date"])]

    feature_config = config["features"]
    features = build_features(
        prices,
        momentum_windows=feature_config["momentum_windows"],
        volatility_windows=feature_config["volatility_windows"],
        trend_windows=feature_config["trend_windows"],
        forward_return_horizon=feature_config["forward_return_horizon"],
    )
    if features.empty:
        raise ValueError("Feature matrix is empty; check price history length and config windows.")

    results = run_all_strategies(prices, features, config)
    returns = align_strategy_returns(results)
    metrics = results_to_metrics(results)
    equity = equity_curves(results)
    weights = weights_to_frame(results)
    metadata = _build_metadata(config, prices, returns)

    metrics_dir = ROOT / "reports" / "metrics"
    figures_dir = ROOT / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(metrics_dir / "backtest_metrics.csv")
    equity.to_csv(metrics_dir / "equity_curves.csv", index_label="date")
    weights.to_csv(metrics_dir / "portfolio_weights.csv", index=False)
    metadata.to_csv(metrics_dir / "backtest_metadata.csv", index=False)

    plot_equity_curves(equity, figures_dir / "equity_curve.png")
    plot_drawdowns(equity, figures_dir / "drawdown.png")
    plot_rolling_sharpe(returns, figures_dir / "rolling_sharpe.png")
    plot_allocation_heatmap(weights, figures_dir / "allocation_heatmap.png")

    display_columns = [
        "cumulative_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
        "initial_turnover",
    ]
    print(metrics[display_columns].round(4).to_string())
    print(
        "Evaluation period: "
        f"{metadata.loc[0, 'evaluation_start_date']} to {metadata.loc[0, 'evaluation_end_date']} "
        f"(price data: {metadata.loc[0, 'data_start_date']} to {metadata.loc[0, 'data_end_date']})"
    )


def _load_or_download_prices(config: dict) -> pd.DataFrame:
    tickers = config["data"]["tickers"]
    raw_path = ROOT / "data" / "raw" / "prices.csv"
    if raw_path.exists():
        return load_prices(raw_path).reindex(columns=tickers)

    try:
        prices = load_legacy_ticker_csvs(ROOT / "data", tickers, config["data"].get("price_column", "Adj Close"))
        save_prices(prices, raw_path)
        return prices.reindex(columns=tickers)
    except FileNotFoundError:
        prices = download_prices(tickers, config["data"]["start_date"], config["data"].get("end_date"))
        save_prices(prices, raw_path)
        return prices.reindex(columns=tickers)


def _validate_or_refresh_data_start(config: dict, prices: pd.DataFrame) -> pd.DataFrame:
    requested_start = pd.Timestamp(config["data"]["start_date"])
    available_start = pd.Timestamp(prices.dropna(how="all").index.min())
    stale_threshold = requested_start + pd.Timedelta(days=30)
    if available_start <= stale_threshold:
        return prices

    message = (
        "WARNING: local price history starts on "
        f"{available_start.date()}, more than 30 calendar days after configured "
        f"start_date {requested_start.date()}."
    )
    print(message)
    if os.environ.get("QUANT_PORTFOLIO_SKIP_REFRESH_DOWNLOAD") == "1":
        print("WARNING: continuing with local data because refresh download is disabled.")
        return prices

    tickers = config["data"]["tickers"]
    raw_path = ROOT / "data" / "raw" / "prices.csv"
    try:
        refreshed = download_prices(tickers, config["data"]["start_date"], config["data"].get("end_date")).reindex(columns=tickers)
    except Exception as exc:
        print(f"WARNING: attempted to refresh prices with yfinance but failed: {exc}")
        print("WARNING: continuing with the later-starting local data; generated results use the available local period.")
        return prices

    refreshed_start = pd.Timestamp(refreshed.dropna(how="all").index.min())
    if refreshed_start < available_start:
        save_prices(refreshed, raw_path)
        print(f"Downloaded refreshed prices starting on {refreshed_start.date()} and updated {raw_path}.")
        return refreshed

    print(
        "WARNING: refreshed download did not extend the available history "
        f"(download starts on {refreshed_start.date()}); continuing with local data."
    )
    return prices


def _build_metadata(config: dict, prices: pd.DataFrame, aligned_returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "configured_start_date": pd.Timestamp(config["data"]["start_date"]).date().isoformat(),
                "data_start_date": pd.Timestamp(prices.index.min()).date().isoformat(),
                "data_end_date": pd.Timestamp(prices.index.max()).date().isoformat(),
                "evaluation_start_date": pd.Timestamp(aligned_returns.index.min()).date().isoformat(),
                "evaluation_end_date": pd.Timestamp(aligned_returns.index.max()).date().isoformat(),
                "forward_return_horizon": int(config["features"]["forward_return_horizon"]),
            }
        ]
    )


if __name__ == "__main__":
    main()
