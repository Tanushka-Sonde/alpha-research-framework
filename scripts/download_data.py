#!/usr/bin/env python
"""Pre-download and cache OHLCV data for the configured universe."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader, DataLoaderConfig  # noqa: E402
from src.pipeline import load_config  # noqa: E402


def main():
    cfg = load_config()
    loader = DataLoader(
        DataLoaderConfig(
            tickers=cfg["data"]["tickers"],
            start_date=cfg["data"]["start_date"],
            end_date=cfg["data"]["end_date"],
            interval=cfg["data"]["interval"],
            cache_dir=cfg["data"]["cache_dir"],
            use_synthetic_if_offline=cfg["data"]["use_synthetic_if_offline"],
        )
    )
    universe = loader.load_universe()
    for ticker, df in universe.items():
        print(f"{ticker}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")


if __name__ == "__main__":
    main()
