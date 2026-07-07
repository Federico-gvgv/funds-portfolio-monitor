from __future__ import annotations

import pandas as pd


DEFAULT_MOMENTUM_WINDOWS = (21, 63, 126, 252)
DEFAULT_VOLATILITY_WINDOWS = (20, 60)
DEFAULT_TREND_WINDOWS = (50, 200)


def build_features(
    prices: pd.DataFrame,
    momentum_windows: list[int] | tuple[int, ...] = DEFAULT_MOMENTUM_WINDOWS,
    volatility_windows: list[int] | tuple[int, ...] = DEFAULT_VOLATILITY_WINDOWS,
    trend_windows: list[int] | tuple[int, ...] = DEFAULT_TREND_WINDOWS,
    forward_return_horizon: int = 21,
) -> pd.DataFrame:
    """Build leakage-safe cross-sectional features and forward-return target."""
    prices = prices.sort_index().copy()
    returns = prices.pct_change()

    frames: list[pd.DataFrame] = []
    for ticker in prices.columns:
        price = prices[ticker]
        ret = returns[ticker]
        frame = pd.DataFrame(index=prices.index)
        frame["date"] = frame.index
        frame["ticker"] = ticker
        frame["return_1d"] = ret

        for window in momentum_windows:
            frame[f"momentum_{window}"] = price / price.shift(window) - 1.0
        for window in volatility_windows:
            frame[f"volatility_{window}"] = ret.rolling(window).std()
        for window in trend_windows:
            moving_average = price.rolling(window).mean()
            frame[f"trend_{window}"] = price / moving_average - 1.0

        rolling_high = price.rolling(252).max()
        frame["drawdown_252"] = price / rolling_high - 1.0
        frame[f"forward_return_{forward_return_horizon}"] = (
            price.shift(-forward_return_horizon) / price - 1.0
        )
        frames.append(frame)

    tidy = pd.concat(frames, ignore_index=True)
    feature_columns = [
        "return_1d",
        *[f"momentum_{window}" for window in momentum_windows],
        *[f"volatility_{window}" for window in volatility_windows],
        *[f"trend_{window}" for window in trend_windows],
        "drawdown_252",
        f"forward_return_{forward_return_horizon}",
    ]
    tidy = tidy.dropna(subset=feature_columns)
    tidy["date"] = pd.to_datetime(tidy["date"])
    return tidy.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_column_names(features: pd.DataFrame, target_column: str = "forward_return_21") -> list[str]:
    excluded = {"date", "ticker", target_column}
    return [column for column in features.columns if column not in excluded]
