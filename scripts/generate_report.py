from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_portfolio.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown research report.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics_path = ROOT / "reports" / "metrics" / "backtest_metrics.csv"
    metadata_path = ROOT / "reports" / "metrics" / "backtest_metadata.csv"
    metrics_markdown = "_Run `python scripts/run_backtest.py --config configs/backtest.yaml` to generate metrics._"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path, index_col=0)
        metrics_markdown = _to_markdown_table(metrics.round(4))
    metadata = _load_metadata(metadata_path)

    tickers = ", ".join(config["data"]["tickers"])
    cost_bps = config["backtest"]["transaction_cost_bps"]
    horizon = int(config["features"]["forward_return_horizon"])
    report = f"""# Cross-Asset ETF Tactical Allocation Backtester

## Project overview

This project treats ETF allocation as a walk-forward portfolio construction problem rather than a next-price prediction problem. The pipeline emphasizes leakage avoidance, factor-style features, realistic backtesting assumptions, transaction costs, and risk-adjusted evaluation.

## Data universe

The main universe is: {tickers}. Prices are adjusted close series where available, loaded from `data/raw/prices.csv` or downloaded with `yfinance`.

{metadata}

## Methodology

The research pipeline computes daily returns, trailing momentum, realized volatility, moving-average trend, rolling drawdown, and a next-{horizon}-trading-day forward-return target. Features at date `t` use only prices available at or before `t`; labels are future returns used only for supervised training.

Models are retrained chronologically at each monthly rebalance using only rows whose forward-return labels would have been observable by that rebalance date. In practice, the latest training label date is lagged by the forecast horizon. Current rebalance-date features are scored without using current or future target values.

## Feature set

- Daily return
- 21, 63, 126, and 252 trading-day momentum
- 20 and 60 trading-day volatility
- 50 and 200 trading-day trend signals
- Drawdown from the 252 trading-day rolling high
- Next {horizon} trading-day forward return target

## Strategies compared

- SPY buy-and-hold
- Equal-weight ETF universe
- Top-k momentum
- Top-k volatility-adjusted momentum
- Ridge predicted-return top-k
- Gradient boosting predicted-return top-k

## Walk-forward validation

At each monthly rebalance, supervised models train only on historical observations ending at least {horizon} business days before the rebalance date. Current cross-sectional features are scored, the top assets are selected, and weights are applied to the next holding period.

## Transaction costs

The backtest subtracts `{cost_bps}` basis points times one-way turnover at each rebalance.

Initial allocation turnover is reported separately from average recurring turnover for static strategies so buy-and-hold and equal-weight baselines are not presented as if they rebalance every period.

## Performance table

{metrics_markdown}

## Limitations

- ETF-only universe.
- No slippage or market impact beyond a simple transaction cost.
- No borrow, leverage, or shorting.
- No macroeconomic, options, sentiment, or fundamental data.
- Historical backtests are not proof of future performance.
- Results depend on the available data period and should be interpreted as research, not investment advice.
- `yfinance` data may differ from institutional-quality sources.
- Monthly rebalancing and close-to-close execution assumptions are simplified.

## Future work

- Add richer risk controls such as volatility targeting and drawdown stops.
- Add expanding-window versus rolling-window training experiments.
- Compare additional cross-validation designs that preserve chronology.
- Add slippage models and execution timing sensitivity tests.
- Introduce macro and rates features for cross-asset context.
"""
    output_path = ROOT / "reports" / "final_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")

def _to_markdown_table(frame: pd.DataFrame) -> str:
    table = frame.reset_index()
    columns = list(table.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def _load_metadata(path: Path) -> str:
    if not path.exists():
        return "_Run `python scripts/run_backtest.py --config configs/backtest.yaml` to record the actual data and evaluation periods._"

    metadata = pd.read_csv(path).iloc[0]
    return (
        "Generated results use price data from "
        f"`{metadata['data_start_date']}` to `{metadata['data_end_date']}`. "
        "Reported metrics are computed on the common aligned evaluation period from "
        f"`{metadata['evaluation_start_date']}` to `{metadata['evaluation_end_date']}`."
    )


if __name__ == "__main__":
    main()
