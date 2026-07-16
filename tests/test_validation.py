import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation import walk_forward_splits


def _make_dates(n):
    return np.array(
        [np.datetime64("2015-01-01") + np.timedelta64(i, "D") for i in range(n)]
    )


def test_folds_are_chronological_and_non_overlapping():
    dates = _make_dates(500)
    folds = walk_forward_splits(
        dates, train_window_days=200, test_window_days=50, step_days=50, embargo_days=5
    )
    assert len(folds) > 0
    for fold in folds:
        assert fold.train_dates.max() < fold.test_dates.min()
        # embargo gap must exist
        gap_days = (fold.test_dates.min() - fold.train_dates.max()).astype("timedelta64[D]").astype(int)
        assert gap_days >= 5


def test_folds_slide_forward_in_time():
    dates = _make_dates(600)
    folds = walk_forward_splits(
        dates, train_window_days=200, test_window_days=50, step_days=50, embargo_days=5
    )
    for a, b in zip(folds, folds[1:]):
        assert b.train_dates.min() > a.train_dates.min()


def test_raises_when_range_too_short():
    dates = _make_dates(50)
    try:
        walk_forward_splits(dates, train_window_days=200, test_window_days=50, step_days=50, embargo_days=5)
        assert False, "expected ValueError"
    except ValueError:
        pass
