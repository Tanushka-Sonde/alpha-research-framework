"""
Data layer: pulls OHLCV data from Yahoo Finance (via yfinance, free) and caches
it locally as parquet so re-runs don't hammer the API or require internet
every time.

Design note: everything downstream treats a bar at date `t` (the daily
OHLCV row) as "known" only at the *close* of day t. Every feature/label built
on top of this must respect that, which is handled in features.py / labeling.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


def _cache_path(raw_data_dir: str | Path, ticker: str) -> Path:
    return Path(raw_data_dir) / f"{ticker}.parquet"


def download_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    raw_data_dir: str | Path,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download (or load from cache) daily OHLCV data for each ticker.

    Returns a dict {ticker: DataFrame[open, high, low, close, adj_close, volume]}
    indexed by date (tz-naive, sorted ascending).
    """
    import yfinance as yf  # imported lazily so the rest of the package works offline

    Path(raw_data_dir).mkdir(parents=True, exist_ok=True)
    out = {}

    for ticker in tickers:
        cache_file = _cache_path(raw_data_dir, ticker)

        if cache_file.exists() and not force_refresh:
            df = pd.read_parquet(cache_file)
            logger.info(f"{ticker}: loaded {len(df)} rows from cache")
        else:
            logger.info(f"{ticker}: downloading {start_date} -> {end_date}")
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )
            if raw.empty:
                logger.warning(f"{ticker}: no data returned, skipping")
                continue

            # yfinance can return a MultiIndex column frame for a single ticker
            # in some versions; flatten defensively.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )[["open", "high", "low", "close", "adj_close", "volume"]]
            df.index.name = "date"
            df = df.sort_index()
            df.to_parquet(cache_file)
            logger.info(f"{ticker}: cached {len(df)} rows -> {cache_file}")

        out[ticker] = df

    if not out:
        raise RuntimeError(
            "No data was downloaded for any ticker. Check your internet connection "
            "or ticker symbols."
        )

    return out


def align_calendar(ohlcv: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Reindex all tickers onto the union of trading dates seen across the
    universe, forward-filling small gaps (e.g. a single-exchange holiday)
    but never filling from the future. This keeps the panel rectangular for
    the walk-forward loop without introducing look-ahead bias.
    """
    all_dates = sorted(set().union(*[df.index for df in ohlcv.values()]))
    aligned = {}
    for ticker, df in ohlcv.items():
        aligned[ticker] = df.reindex(all_dates).ffill()
    return aligned
