# Cross-Asset ETF Tactical Allocation Backtester

## Project overview

This project treats ETF allocation as a walk-forward portfolio construction problem rather than a next-price prediction problem. The pipeline emphasizes leakage avoidance, factor-style features, realistic backtesting assumptions, transaction costs, and risk-adjusted evaluation.

## Data universe

The main universe is: SPY, QQQ, IWM, EFA, EEM, TLT, GLD. Prices are adjusted close series where available, loaded from `data/raw/prices.csv` or downloaded with `yfinance`.

Generated results use price data from `2010-01-04` to `2026-07-07`. Reported metrics are computed on the common aligned evaluation period from `2016-03-01` to `2026-05-29`.

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
| SPY Buy & Hold | 2.3661 | 0.1841 | 0.1662 | 1.1003 | 1.4295 | -0.2318 | 0.7941 | 0.5586 | 0.0 | 0.5 | 3.3661 |
| Equal Weight Universe | 1.7403 | 0.1507 | 0.1287 | 1.1552 | 1.6565 | -0.2267 | 0.6647 | 0.563 | 0.0 | 0.5 | 2.7403 |
| Top-k Momentum | 2.0394 | 0.1674 | 0.1448 | 1.1414 | 1.5477 | -0.2417 | 0.6926 | 0.5552 | 0.1938 | 0.5 | 3.0394 |
| Top-k Vol-Adjusted Momentum | 1.6648 | 0.1462 | 0.1441 | 1.0193 | 1.372 | -0.2316 | 0.6312 | 0.5503 | 0.2481 | 0.5 | 2.6648 |
| Ridge Predicted Return | 1.3518 | 0.1264 | 0.1687 | 0.7899 | 1.08 | -0.2777 | 0.4553 | 0.5448 | 0.3876 | 0.5 | 2.3518 |
| Gradient Boosting Predicted Return | 1.729 | 0.15 | 0.1604 | 0.9514 | 1.2942 | -0.2145 | 0.6992 | 0.5486 | 0.4806 | 0.5 | 2.729 |

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
