# Cross-Asset ETF Tactical Allocation Backtester

## Project overview

This project treats ETF allocation as a walk-forward portfolio construction problem rather than a next-price prediction problem. The pipeline emphasizes leakage avoidance, factor-style features, realistic backtesting assumptions, transaction costs, and risk-adjusted evaluation.

## Data universe

The main universe is: SPY, QQQ, IWM, EFA, EEM, TLT, GLD. Prices are adjusted close series where available, loaded from `data/raw/prices.csv` or downloaded with `yfinance`.

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

The backtest subtracts `10` basis points times one-way turnover at each rebalance.

## Performance table

| strategy | cumulative_return | cagr | annualized_volatility | sharpe_ratio | sortino_ratio | max_drawdown | calmar_ratio | hit_rate | average_turnover | final_equity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY Buy & Hold | 0.5468 | 0.2453 | 0.1639 | 1.4205 | 1.7905 | -0.1876 | 1.308 | 0.5928 | 0.5 | 1.5468 |
| Equal Weight Universe | 0.4959 | 0.2245 | 0.1277 | 1.6507 | 2.2945 | -0.1276 | 1.7602 | 0.5828 | 0.5 | 1.4959 |
| Top-k Momentum | 0.4092 | 0.2964 | 0.1463 | 1.8484 | 2.5383 | -0.139 | 2.1317 | 0.5646 | 0.1354 | 1.4092 |
| Top-k Vol-Adjusted Momentum | 0.3741 | 0.2719 | 0.1454 | 1.7273 | 2.3666 | -0.1366 | 1.9905 | 0.5736 | 0.2396 | 1.3741 |
| Ridge Predicted Return | 0.4008 | 0.2905 | 0.1517 | 1.7575 | 2.3958 | -0.1335 | 2.1761 | 0.5886 | 0.2812 | 1.4008 |
| Gradient Boosting Predicted Return | 0.5908 | 0.4209 | 0.1246 | 2.8825 | 4.2704 | -0.0648 | 6.4925 | 0.5856 | 0.3646 | 1.5908 |

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
