import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import risk_metrics as rm


def test_sharpe_zero_vol_is_nan():
    returns = pd.Series([0.0] * 100)
    assert np.isnan(rm.sharpe_ratio(returns))


def test_max_drawdown_on_monotonic_series_is_zero():
    equity = pd.Series(np.linspace(100, 200, 50))
    assert rm.max_drawdown(equity) == 0.0


def test_max_drawdown_detects_known_drop():
    equity = pd.Series([100, 120, 90, 95, 130])
    # peak 120 -> trough 90 => -25%
    assert abs(rm.max_drawdown(equity) - (-0.25)) < 1e-9


def test_hit_rate_basic():
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.0])
    assert rm.hit_rate(returns) == 0.5  # 2 positive out of 4 nonzero
