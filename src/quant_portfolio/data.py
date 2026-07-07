from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def download_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to download market data.") from exc

    tickers = list(tickers)
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise ValueError("No prices were downloaded.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            raise ValueError("Downloaded data does not include close prices.")
    else:
        column = "Adj Close" if "Adj Close" in data.columns else "Close"
        prices = data[[column]].rename(columns={column: tickers[0]})

    prices = prices.reindex(columns=tickers)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return _clean_prices(prices)


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a prices CSV with dates in the first column and tickers as columns."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.index.name = "date"
    return _clean_prices(prices)


def save_prices(prices: pd.DataFrame, path: str | Path) -> None:
    """Save prices to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.sort_index().to_csv(path, index_label="date")


def load_legacy_ticker_csvs(
    folder: str | Path,
    tickers: Iterable[str],
    price_column: str = "Adj Close",
) -> pd.DataFrame:
    """Load one CSV per ticker from the original project data folder."""
    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = Path(folder) / f"{ticker}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["Date"])
        column = price_column if price_column in frame.columns else "Close"
        if column not in frame.columns:
            continue
        series = frame.set_index("Date")[column].rename(ticker)
        frames[ticker] = series

    if not frames:
        raise FileNotFoundError(f"No per-ticker CSV files found in {folder}.")
    prices = pd.concat(frames.values(), axis=1).sort_index()
    return _clean_prices(prices)


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")
    return prices.ffill()
