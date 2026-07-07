# Cross-Asset ETF Tactical Allocation Backtester

This project is a reproducible quant-research pipeline for cross-asset ETF tactical allocation. It compares simple baselines, factor-style ranking rules, and lightweight supervised ML models under chronological walk-forward validation.

This project treats ETF allocation as a walk-forward portfolio construction problem rather than a next-price prediction problem. The pipeline emphasizes leakage avoidance, factor-style features, realistic backtesting assumptions, transaction costs, and risk-adjusted evaluation.

## Why this is not a naive price forecaster

The main workflow does not train an LSTM to predict next prices. Instead, it builds trailing momentum, volatility, trend, and drawdown features, predicts or ranks next 21-trading-day returns, rebalances monthly, and evaluates complete portfolios after turnover and transaction costs.

Legacy LSTM price-forecasting code has been moved to `legacy/lstm_price_forecaster/` and is not part of the active research path.

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
5. Train supervised models only on rows before each rebalance date.
6. Select long-only top-k portfolios and apply weights to the next holding period.
7. Subtract transaction costs as basis points times one-way turnover.
8. Evaluate portfolio-level risk and return metrics.

## Strategies

- `SPY Buy & Hold`
- `Equal Weight Universe`
- `Top-k Momentum`
- `Top-k Vol-Adjusted Momentum`
- `Ridge Predicted Return`
- `Gradient Boosting Predicted Return`

## Metrics

The report includes cumulative return, CAGR, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, hit rate, average turnover, and final equity.

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
- `yfinance` data may differ from institutional-quality sources.
- Monthly rebalancing and close-to-close execution assumptions are simplified.

## Future work

- Add volatility targeting and portfolio-level risk constraints.
- Compare expanding-window and rolling-window training regimes.
- Add slippage and execution timing sensitivity analysis.
- Add macro, rates, valuation, or carry features.
- Add richer reporting for regime-specific performance.
