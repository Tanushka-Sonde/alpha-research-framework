"""
Data loader
===========
Pulls raw OHLCV bars for a universe of tickers and caches them locally as
parquet so repeated runs don't hammer the data source.

Design notes:
- We cache per-ticker, per-interval. Cache is invalidated by date range,
  not by content hash — simple and good enough for a research repo.
- If yfinance can't reach the network (common in sandboxed / offline dev
  environments), we fall back to a synthetic multi-regime price generator.
  This keeps the *entire pipeline* runnable end-to-end without internet,
  which is useful for testing, CI, and demos. Swap back to real data by
  just having a working internet connection — no code changes needed.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class DataLoaderConfig:
    tickers: list
    start_date: str
    end_date: Optional[str]
    interval: str = "1d"
    cache_dir: str = "data_cache"
    use_synthetic_if_offline: bool = True


class DataLoader:
    def __init__(self, cfg: DataLoaderConfig):
        self.cfg = cfg
        Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load_universe(self) -> dict[str, pd.DataFrame]:
        """Returns {ticker: OHLCV DataFrame indexed by date}."""
        out = {}
        for ticker in self.cfg.tickers:
            out[ticker] = self.load_single(ticker)
        return out

    def load_single(self, ticker: str) -> pd.DataFrame:
        cache_path = self._cache_path(ticker)
        if cache_path.exists():
            logger.info("Loading %s from cache (%s)", ticker, cache_path)
            df = pd.read_parquet(cache_path)
            return self._validate(df, ticker)

        df = self._download(ticker)
        df = self._validate(df, ticker)
        df.to_parquet(cache_path)
        return df

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _cache_path(self, ticker: str) -> Path:
        end = self.cfg.end_date or "latest"
        fname = f"{ticker}_{self.cfg.start_date}_{end}_{self.cfg.interval}.parquet"
        return Path(self.cfg.cache_dir) / fname

    def _download(self, ticker: str) -> pd.DataFrame:
        try:
            import yfinance as yf

            raw = yf.download(
                ticker,
                start=self.cfg.start_date,
                end=self.cfg.end_date,
                interval=self.cfg.interval,
                auto_adjust=True,
                progress=False,
            )
            if raw is None or raw.empty:
                raise ValueError("yfinance returned no rows")

            # yfinance sometimes returns MultiIndex columns even for one ticker
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw.index.name = "date"
            logger.info("Downloaded %s: %d rows from yfinance", ticker, len(raw))
            return raw[REQUIRED_COLUMNS]

        except Exception as exc:  # noqa: BLE001 - broad on purpose, many failure modes offline
            if not self.cfg.use_synthetic_if_offline:
                raise
            logger.warning(
                "Falling back to SYNTHETIC data for %s (reason: %s). "
                "This is fine for testing the pipeline, but train/evaluate on "
                "real data before drawing any conclusions.",
                ticker, exc,
            )
            return self._synthetic(ticker)

    def _synthetic(self, ticker: str) -> pd.DataFrame:
        """
        Multi-regime synthetic OHLCV generator. Not random-walk noise —
        stitches together low-vol/trending and high-vol/mean-reverting
        regimes so downstream regime analysis (in the dashboard) has
        something real to show, and so a model can't just memorize one
        constant-vol process.
        """
        rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
        start = pd.Timestamp(self.cfg.start_date)
        end = pd.Timestamp(self.cfg.end_date) if self.cfg.end_date else pd.Timestamp.today()
        dates = pd.bdate_range(start, end)
        n = len(dates)

        regime_len = rng.integers(40, 120)
        prices = [100.0]
        vols, drifts = [], []
        i = 0
        while i < n:
            length = min(regime_len, n - i)
            regime = rng.choice(["trend_up", "trend_down", "mean_revert", "high_vol_chop"])
            if regime == "trend_up":
                drift, vol = 0.0006, 0.010
            elif regime == "trend_down":
                drift, vol = -0.0005, 0.012
            elif regime == "mean_revert":
                drift, vol = 0.0000, 0.008
            else:
                drift, vol = 0.0002, 0.022
            drifts += [drift] * length
            vols += [vol] * length
            i += length
            regime_len = rng.integers(40, 120)

        drifts = np.array(drifts[:n])
        vols = np.array(vols[:n])
        shocks = rng.normal(drifts, vols)
        log_prices = np.log(prices[0]) + np.cumsum(shocks)
        close = np.exp(log_prices)

        intraday_range = np.abs(rng.normal(0, vols * 0.6, n))
        high = close * (1 + intraday_range)
        low = close * (1 - intraday_range)
        open_ = low + (high - low) * rng.uniform(0, 1, n)
        volume = rng.lognormal(mean=15, sigma=0.4, size=n)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        df.index.name = "date"
        return df

    def _validate(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{ticker}: missing columns {missing}")
        df = df[REQUIRED_COLUMNS].copy()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # Drop rows that are fully NaN (holidays/gaps sometimes leak through)
        df = df.dropna(how="all")
        # Forward-fill isolated single-day gaps only, never bulk-fill —
        # bulk-filling would fabricate price history.
        n_nans_before = df.isna().sum().sum()
        df = df.ffill(limit=1)
        if df.isna().sum().sum() > 0:
            df = df.dropna()
        if n_nans_before:
            logger.info("%s: forward-filled/dropped %d NaN cells", ticker, n_nans_before)
        return df
