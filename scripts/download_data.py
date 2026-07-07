from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_portfolio.config import load_config
from quant_portfolio.data import download_prices, save_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Download adjusted close prices.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--output", default="data/raw/prices.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    prices = download_prices(
        tickers=config["data"]["tickers"],
        start_date=config["data"]["start_date"],
        end_date=config["data"].get("end_date"),
    )
    save_prices(prices, args.output)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} tickers to {args.output}")


if __name__ == "__main__":
    main()
