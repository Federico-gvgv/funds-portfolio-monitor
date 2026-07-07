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
    metrics_markdown = "_Run `python scripts/run_backtest.py --config configs/backtest.yaml` to generate metrics._"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path, index_col=0)
        metrics_markdown = _to_markdown_table(metrics.round(4))

    tickers = ", ".join(config["data"]["tickers"])
    cost_bps = config["backtest"]["transaction_cost_bps"]
    report = f"""# Cross-Asset ETF Tactical Allocation Backtester

## Project overview

This project treats ETF allocation as a walk-forward portfolio construction problem rather than a next-price prediction problem. The pipeline emphasizes leakage avoidance, factor-style features, realistic backtesting assumptions, transaction costs, and risk-adjusted evaluation.

## Data universe

The main universe is: {tickers}. Prices are adjusted close series where available, loaded from `data/raw/prices.csv` or downloaded with `yfinance`.

## Methodology

The research pipeline computes daily returns, trailing momentum, realized volatility, moving-average trend, rolling drawdown, and a next-21-trading-day forward-return target. Features at date `t` use only prices available at or before `t`; labels are future returns used only for supervised training.

Models are retrained chronologically at each monthly rebalance using rows before the rebalance date. The test period is not used for hyperparameter selection.

## Feature set

- Daily return
- 21, 63, 126, and 252 trading-day momentum
- 20 and 60 trading-day volatility
- 50 and 200 trading-day trend signals
- Drawdown from the 252 trading-day rolling high
- Next 21 trading-day forward return target

## Strategies compared

- SPY buy-and-hold
- Equal-weight ETF universe
- Top-k momentum
- Top-k volatility-adjusted momentum
- Ridge predicted-return top-k
- Gradient boosting predicted-return top-k

## Walk-forward validation

At each monthly rebalance, supervised models train only on historical observations before the rebalance date. Current cross-sectional features are scored, the top assets are selected, and weights are applied to the next holding period.

## Transaction costs

The backtest subtracts `{cost_bps}` basis points times one-way turnover at each rebalance.

## Performance table

{metrics_markdown}

## Limitations

- ETF-only universe.
- No slippage or market impact beyond a simple transaction cost.
- No borrow, leverage, or shorting.
- No macroeconomic, options, sentiment, or fundamental data.
- Historical backtests are not proof of future performance.
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


if __name__ == "__main__":
    main()
