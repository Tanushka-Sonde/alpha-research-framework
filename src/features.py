"""
Feature engineering — everything here is computed using only information
available up to and including the current bar's close. That's what "point in
time" means: no centered rolling windows, no using tomorrow's data to
normalize today's value, no fitting scalers on the full history before a
split.

Every function operates on a single ticker's OHLCV frame and returns a
DataFrame of features aligned on the same date index. build_feature_panel()
stitches these into a long panel indexed by (date, ticker), which is the
shape the rest of the pipeline expects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    """Simple trailing return over `window` trading days, known as of today."""
    return close.pct_change(window)


def rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Realized volatility (std of daily log returns) over a trailing window."""
    log_ret = np.log(close).diff()
    return log_ret.rolling(window, min_periods=window).std()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder RSI, trailing-only."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """MACD line and signal line, both trailing-only (EWMs)."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line


def bollinger_bandwidth(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """(Upper band - lower band) / middle band — a trailing volatility-regime proxy."""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return (upper - lower) / mid


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Trailing z-score of volume, flags unusual participation."""
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std()
    return (volume - mean) / std.replace(0, np.nan)


def build_features_single_ticker(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build the full feature set for one ticker's OHLCV frame."""
    close = df["adj_close"]
    feats = pd.DataFrame(index=df.index)

    for w in cfg["return_windows"]:
        feats[f"ret_{w}d"] = rolling_return(close, w)

    for w in cfg["vol_windows"]:
        feats[f"vol_{w}d"] = rolling_volatility(close, w)

    feats["rsi"] = rsi(close, cfg["rsi_window"])

    macd_line, signal_line = macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    feats["macd"] = macd_line
    feats["macd_signal"] = signal_line
    feats["macd_hist"] = macd_line - signal_line

    feats["bb_bandwidth"] = bollinger_bandwidth(close, cfg["bollinger_window"])
    feats["volume_zscore"] = volume_zscore(df["volume"], cfg["volume_zscore_window"])

    # Store raw close too — needed downstream for labeling & backtest pricing,
    # not used directly as a model feature (it's non-stationary).
    feats["_close"] = close
    return feats


def build_feature_panel(ohlcv: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """
    Build features for every ticker and stack into a long panel with a
    (date, ticker) MultiIndex. This is the shape used by labeling,
    validation, and the model training loop.
    """
    frames = []
    for ticker, df in ohlcv.items():
        f = build_features_single_ticker(df, cfg)
        f["ticker"] = ticker
        frames.append(f)

    panel = pd.concat(frames)
    panel = panel.set_index("ticker", append=True)
    panel.index.names = ["date", "ticker"]
    return panel.sort_index()


FEATURE_COLUMNS = None  # populated lazily by callers via `get_feature_columns`


def get_feature_columns(panel: pd.DataFrame) -> list[str]:
    """
    Model-usable feature columns: excludes internal bookkeeping columns
    (prefixed with `_`, e.g. `_close`) AND excludes `label` if present.
    Critical: `label` is the training target — including it as a feature
    would let the model "predict" the target from itself, producing
    trivially perfect (and completely fake) backtest results. This is a
    safety net regardless of when in the pipeline you call this function.
    """
    return [c for c in panel.columns if not c.startswith("_") and c != "label"]
