"""
Feature engineering
====================
Every feature here is computed using only information available AT OR
BEFORE the row's timestamp (close-to-close data up to and including day t).
The label (forward return) is the only thing that looks into the future,
and it lives in a clearly separate function so it's impossible to
accidentally feed it back in as a feature.

Why this matters: the single most common way student projects get torn
apart in interviews is silently leaking t+1..t+H information into a
feature at time t (e.g. computing an indicator on a centered rolling
window, or using `.shift(-1)` in a helper meant for something else).
Every rolling computation below is either causal by construction
(pandas `.rolling()` looks backward already) or explicitly commented
where care was needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    return_horizons: list = field(default_factory=lambda: [1, 5, 10, 21])
    vol_windows: list = field(default_factory=lambda: [5, 10, 21, 63])
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    zscore_window: int = 21
    volume_windows: list = field(default_factory=lambda: [5, 21])


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing (an EMA variant) — causal, uses only past+current bar
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd_line": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def build_features(ohlcv: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    ohlcv: DataFrame with columns [open, high, low, close, volume], indexed by date.
    Returns a feature DataFrame aligned to the same index (with NaN warmup rows
    still present — caller decides how to trim).
    """
    df = ohlcv.copy()
    close = df["close"]
    log_ret_1 = np.log(close / close.shift(1))

    feats = pd.DataFrame(index=df.index)

    # --- Momentum / return features (causal: shift(h) uses only past prices) ---
    for h in cfg.return_horizons:
        feats[f"ret_{h}d"] = close.pct_change(h)
        feats[f"logret_{h}d"] = np.log(close / close.shift(h))

    # --- Realized volatility over multiple windows (rolling std of daily log returns) ---
    for w in cfg.vol_windows:
        feats[f"vol_{w}d"] = log_ret_1.rolling(w, min_periods=w).std() * np.sqrt(252)

    # --- Volatility regime ratio: short vol vs long vol (captures vol clustering) ---
    if len(cfg.vol_windows) >= 2:
        short_w, long_w = min(cfg.vol_windows), max(cfg.vol_windows)
        feats["vol_ratio_short_long"] = feats[f"vol_{short_w}d"] / feats[f"vol_{long_w}d"].replace(0, np.nan)

    # --- RSI ---
    feats[f"rsi_{cfg.rsi_window}"] = _rsi(close, cfg.rsi_window)

    # --- MACD ---
    macd_df = _macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    feats = feats.join(macd_df)
    feats["macd_hist_norm"] = macd_df["macd_hist"] / close  # scale-free version

    # --- Bollinger Bands: position of price within the band, and band width ---
    bb_mid = close.rolling(cfg.bollinger_window, min_periods=cfg.bollinger_window).mean()
    bb_std = close.rolling(cfg.bollinger_window, min_periods=cfg.bollinger_window).std()
    bb_upper = bb_mid + cfg.bollinger_std * bb_std
    bb_lower = bb_mid - cfg.bollinger_std * bb_std
    feats["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    feats["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    # --- Rolling z-score of price vs its own recent mean (mean-reversion signal) ---
    roll_mean = close.rolling(cfg.zscore_window, min_periods=cfg.zscore_window).mean()
    roll_std = close.rolling(cfg.zscore_window, min_periods=cfg.zscore_window).std()
    feats[f"zscore_{cfg.zscore_window}"] = (close - roll_mean) / roll_std.replace(0, np.nan)

    # --- Volume features: relative volume + volume trend ---
    vol = df["volume"]
    for w in cfg.volume_windows:
        avg_vol = vol.rolling(w, min_periods=w).mean()
        feats[f"rel_volume_{w}d"] = vol / avg_vol.replace(0, np.nan)
    feats["volume_trend"] = vol.rolling(cfg.volume_windows[0]).mean() / vol.rolling(
        cfg.volume_windows[-1]
    ).mean().replace(0, np.nan)

    # --- Intraday range / gap features (causal — same-day OHLC is known by close) ---
    feats["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]
    feats["co_gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # --- Candle body location within the day's range (0=closed at low, 1=closed at high) ---
    day_range = (df["high"] - df["low"]).replace(0, np.nan)
    feats["close_position_in_range"] = (df["close"] - df["low"]) / day_range

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


def make_labels(ohlcv: pd.DataFrame, horizon: int, target_type: str = "return") -> pd.Series:
    """
    Forward-looking target — ONLY function in this module allowed to look ahead.
    label[t] = return from close[t] to close[t+horizon].

    Rows where t+horizon exceeds the available data are NaN and must be
    dropped before training (they're "unknown future" by construction — if
    you fill them with anything, you are fabricating a label).
    """
    close = ohlcv["close"]
    fwd_ret = close.shift(-horizon) / close - 1.0
    if target_type == "return":
        return fwd_ret.rename(f"fwd_ret_{horizon}d")
    elif target_type == "direction":
        return (fwd_ret > 0).astype(float).rename(f"fwd_dir_{horizon}d")
    else:
        raise ValueError(f"Unknown target_type: {target_type}")


def assemble_dataset(
    ohlcv: pd.DataFrame, feat_cfg: FeatureConfig, label_horizon: int, target_type: str = "return"
) -> pd.DataFrame:
    """
    Joins features + label, drops warmup NaNs (feature side) and the final
    `label_horizon` rows (label side, unknown future). Returns a single
    tidy DataFrame: feature columns + a 'label' column, indexed by date.
    """
    feats = build_features(ohlcv, feat_cfg)
    label = make_labels(ohlcv, label_horizon, target_type)
    out = feats.copy()
    out["label"] = label
    # Drop feature-warmup NaNs AND the last `label_horizon` rows (label NaNs)
    out = out.dropna()
    return out
