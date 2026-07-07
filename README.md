# Cross-Asset ETF Tactical Allocation Backtester

This project is a reproducible quant-research pipeline for cross-asset ETF tactical allocation. It compares simple baselines, factor-style ranking rules, and lightweight supervised ML models under chronological walk-forward validation.

This project treats ETF allocation as a walk-forward portfolio construction problem rather than a next-price prediction problem. The pipeline emphasizes leakage avoidance, factor-style features, realistic backtesting assumptions, transaction costs, and risk-adjusted evaluation.

## Research approach

The workflow builds trailing momentum, volatility, trend, and drawdown features, then ranks or predicts next 21-trading-day returns across the ETF universe. Portfolios are rebalanced monthly, evaluated after transaction costs, and compared against transparent allocation baselines.

Generated results record both the actual price-data period and the aligned evaluation period used for metrics. The current generated artifacts use price data from `2020-12-07` to `2025-12-05`, with reported metrics aligned on `2024-02-01` to `2025-11-28`. The current bundled `data/raw/prices.csv` starts after the configured `2010-01-01` start date, so the run script warns clearly and attempts a `yfinance` refresh before continuing with local data if refresh is unavailable.

An archived version of the earlier experimental workflow is kept in `legacy/lstm_price_forecaster/` for reference.

## Universe

The default ETF universe is:

- `SPY`
- `QQQ`
- `IWM`
- `EFA`
- `EEM`
- `TLT`
- `GLD`

Optional single-name equities such as `AAPL` and `MSFT` can be added through `configs/backtest.yaml`, but the main backtest is designed around ETFs.

## Methodology

1. Load adjusted close prices from `data/raw/prices.csv`, existing local ticker CSVs, or `yfinance`.
2. Compute daily returns and leakage-safe trailing features.
3. Build a next 21-trading-day forward-return target for model training.
4. Rebalance monthly after a minimum training history.
5. Train supervised models only on rows whose forward-return labels would have been observable by each rebalance date.
6. Select long-only top-k portfolios and apply weights to the next holding period.
7. Subtract transaction costs as basis points times one-way turnover.
8. Align all strategy daily returns to a common evaluation index before computing portfolio-level risk and return metrics.

## Strategies

- `SPY Buy & Hold`
- `Equal Weight Universe`
- `Top-k Momentum`
- `Top-k Vol-Adjusted Momentum`
- `Ridge Predicted Return`
- `Gradient Boosting Predicted Return`

## Metrics

The report includes cumulative return, CAGR, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, hit rate, average turnover, and final equity.

`average_turnover` is recurring rebalance turnover. Initial allocation turnover is reported separately so static buy-and-hold or equal-weight baselines are not shown as having recurring turnover.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

You can also install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to run

```bash
python scripts/run_backtest.py --config configs/backtest.yaml
python scripts/generate_report.py
pytest
```

The backtest first looks for `data/raw/prices.csv`. If it is missing, it can assemble prices from legacy per-ticker CSV files in `data/`; if those are missing, it downloads adjusted close data with `yfinance`.

## Example output files

- `reports/metrics/backtest_metrics.csv`
- `reports/metrics/equity_curves.csv`
- `reports/metrics/portfolio_weights.csv`
- `reports/figures/equity_curve.png`
- `reports/figures/drawdown.png`
- `reports/figures/rolling_sharpe.png`
- `reports/figures/allocation_heatmap.png`
- `reports/final_report.md`

## Limitations

- ETF-only universe by default.
- No slippage or market impact beyond a simple transaction cost.
- No borrow, leverage, or shorting.
- No macroeconomic or fundamental features.
- Historical backtests are not proof of future performance.
- Results depend on the available data period and should be interpreted as research, not investment advice.
- `yfinance` data may differ from institutional-quality sources.
- Monthly rebalancing and close-to-close execution assumptions are simplified.

## Future work

- Add volatility targeting and portfolio-level risk constraints.
- Compare expanding-window and rolling-window training regimes.
- Add slippage and execution timing sensitivity analysis.
- Add macro, rates, valuation, or carry features.
- Add richer reporting for regime-specific performance.
