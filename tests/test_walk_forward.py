import numpy as np
import pandas as pd
import pytest

from src.validation.walk_forward import PurgedWalkForward, WalkForwardConfig


def _make_index(n=1000):
    return pd.bdate_range("2018-01-01", periods=n)


def test_folds_are_strictly_time_ordered():
    idx = _make_index()
    cfg = WalkForwardConfig(n_splits=5, min_train_size=400, test_size=50, embargo_days=5, label_horizon=5)
    splitter = PurgedWalkForward(cfg)
    for train_idx, test_idx in splitter.split(idx):
        assert train_idx.max() < test_idx.min(), "train must precede test"


def test_purge_removes_overlapping_labels():
    idx = _make_index()
    horizon = 10
    cfg = WalkForwardConfig(n_splits=5, min_train_size=400, test_size=50, embargo_days=5, label_horizon=horizon)
    splitter = PurgedWalkForward(cfg)
    for train_idx, test_idx in splitter.split(idx):
        test_start = test_idx.min()
        # No training sample's label window [t, t+horizon] should reach into test_start
        assert train_idx.max() + horizon <= test_start, "purge failed: label overlap into test fold"


def test_embargo_excludes_zone_after_test():
    idx = _make_index()
    cfg = WalkForwardConfig(n_splits=5, min_train_size=400, test_size=50, embargo_days=8, label_horizon=5)
    splitter = PurgedWalkForward(cfg)
    folds = list(splitter.split(idx))
    assert len(folds) >= 2
    # For the second fold onward, training data should never include the
    # embargo zone trailing an earlier test fold
    _, first_test = folds[0]
    embargo_zone = set(range(first_test.max() + 1, first_test.max() + 1 + cfg.embargo_days))
    for train_idx, _ in folds[1:]:
        assert embargo_zone.isdisjoint(set(train_idx.tolist()))


def test_raises_when_not_enough_data():
    idx = _make_index(n=100)
    cfg = WalkForwardConfig(n_splits=5, min_train_size=400, test_size=50, embargo_days=5, label_horizon=5)
    splitter = PurgedWalkForward(cfg)
    with pytest.raises(ValueError):
        list(splitter.split(idx))
