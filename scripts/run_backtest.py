from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_portfolio.backtest import equity_curves, results_to_metrics, run_all_strategies, weights_to_frame
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
    metrics = results_to_metrics(results)
    equity = equity_curves(results)
    weights = weights_to_frame(results)
    returns = pd.concat({name: result.returns for name, result in results.items()}, axis=1)

    metrics_dir = ROOT / "reports" / "metrics"
    figures_dir = ROOT / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(metrics_dir / "backtest_metrics.csv")
    equity.to_csv(metrics_dir / "equity_curves.csv", index_label="date")
    weights.to_csv(metrics_dir / "portfolio_weights.csv", index=False)

    plot_equity_curves(equity, figures_dir / "equity_curve.png")
    plot_drawdowns(equity, figures_dir / "drawdown.png")
    plot_rolling_sharpe(returns, figures_dir / "rolling_sharpe.png")
    plot_allocation_heatmap(weights, figures_dir / "allocation_heatmap.png")

    display_columns = ["cumulative_return", "cagr", "annualized_volatility", "sharpe_ratio", "max_drawdown", "average_turnover"]
    print(metrics[display_columns].round(4).to_string())


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


if __name__ == "__main__":
    main()
