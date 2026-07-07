# Cross-Asset ETF Tactical Allocation Backtester

## Project overview

This project treats ETF allocation as a walk-forward portfolio construction problem rather than a next-price prediction problem. The pipeline emphasizes leakage avoidance, factor-style features, realistic backtesting assumptions, transaction costs, and risk-adjusted evaluation.

## Data universe

The main universe is: SPY, QQQ, IWM, EFA, EEM, TLT, GLD. Prices are adjusted close series where available, loaded from `data/raw/prices.csv` or downloaded with `yfinance`.

Generated results use price data from `2020-12-07` to `2025-12-05`. Reported metrics are computed on the common aligned evaluation period from `2024-02-01` to `2025-11-28`.

## Methodology

The research pipeline computes daily returns, trailing momentum, realized volatility, moving-average trend, rolling drawdown, and a next-21-trading-day forward-return target. Features at date `t` use only prices available at or before `t`; labels are future returns used only for supervised training.

Models are retrained chronologically at each monthly rebalance using only rows whose forward-return labels would have been observable by that rebalance date. In practice, the latest training label date is lagged by the forecast horizon. Current rebalance-date features are scored without using current or future target values.

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

At each monthly rebalance, supervised models train only on historical observations ending at least 21 business days before the rebalance date. Current cross-sectional features are scored, the top assets are selected, and weights are applied to the next holding period.

## Transaction costs

The backtest subtracts `10` basis points times one-way turnover at each rebalance.

Initial allocation turnover is reported separately from average recurring turnover for static strategies so buy-and-hold and equal-weight baselines are not presented as if they rebalance every period.

## Performance table

| strategy | cumulative_return | cagr | annualized_volatility | sharpe_ratio | sortino_ratio | max_drawdown | calmar_ratio | hit_rate | average_turnover | initial_turnover | final_equity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY Buy & Hold | 0.3721 | 0.2705 | 0.1827 | 1.4013 | 1.7714 | -0.1876 | 1.4422 | 0.5946 | 0.0 | 0.5 | 1.3721 |
| Equal Weight Universe | 0.3195 | 0.2335 | 0.1364 | 1.607 | 2.2458 | -0.1276 | 1.8303 | 0.5916 | 0.0 | 0.5 | 1.3195 |
| Top-k Momentum | 0.4092 | 0.2964 | 0.1463 | 1.8484 | 2.5383 | -0.139 | 2.1317 | 0.5646 | 0.1111 | 0.5 | 1.4092 |
| Top-k Vol-Adjusted Momentum | 0.3741 | 0.2719 | 0.1454 | 1.7273 | 2.3666 | -0.1366 | 1.9905 | 0.5736 | 0.2222 | 0.5 | 1.3741 |
| Ridge Predicted Return | 0.3431 | 0.2501 | 0.1487 | 1.5754 | 2.1802 | -0.1335 | 1.8733 | 0.5826 | 0.2444 | 0.5 | 1.3431 |
| Gradient Boosting Predicted Return | 0.3917 | 0.2842 | 0.1186 | 2.1682 | 3.1529 | -0.0787 | 3.6129 | 0.5646 | 0.4444 | 0.5 | 1.3917 |

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
