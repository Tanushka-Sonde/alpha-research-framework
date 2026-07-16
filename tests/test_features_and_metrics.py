import numpy as np
import pandas as pd

from src.data.features import FeatureConfig, assemble_dataset, make_labels
from src.reporting import metrics


def _fake_ohlcv(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = low + (high - low) * rng.uniform(0, 1, n)
    volume = rng.lognormal(15, 0.3, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def test_label_matches_manual_forward_return():
    ohlcv = _fake_ohlcv()
    horizon = 5
    label = make_labels(ohlcv, horizon, "return")
    manual = ohlcv["close"].shift(-horizon) / ohlcv["close"] - 1.0
    pd.testing.assert_series_equal(label, manual.rename(label.name))


def test_last_horizon_rows_are_nan_before_dropna():
    ohlcv = _fake_ohlcv()
    horizon = 7
    label = make_labels(ohlcv, horizon, "return")
    assert label.iloc[-horizon:].isna().all()


def test_assembled_dataset_has_no_nans():
    ohlcv = _fake_ohlcv(n=400)
    feat_cfg = FeatureConfig()
    ds = assemble_dataset(ohlcv, feat_cfg, label_horizon=5)
    assert not ds.isna().any().any()
    assert "label" in ds.columns
    assert len(ds) < len(ohlcv)  # warmup + horizon rows must be trimmed


def test_sharpe_ratio_zero_vol_returns_zero():
    flat = pd.Series([0.0] * 100)
    assert metrics.sharpe_ratio(flat) == 0.0


def test_max_drawdown_is_nonpositive():
    eq = pd.Series([100, 110, 90, 95, 120])
    mdd = metrics.max_drawdown(eq)
    assert mdd <= 0
    assert abs(mdd - (90 / 110 - 1)) < 1e-9
