"""
Walk-forward validation — THE most important file in this project for
avoiding the #1 way student trading projects get shredded in interviews:
lookahead bias from a random train/test split.

Why a random split leaks information here
------------------------------------------
A random split (e.g. sklearn's train_test_split with shuffle=True) scatters
dates from the *entire* history into both train and test. Two problems:

1. Feature leakage via overlapping windows: a rolling-20-day feature at date
   t is a function of dates [t-19, t]. If date t-5 lands in train and date t
   lands in test, the model has effectively "seen" test-period information
   smeared into a training feature window, and vice versa.
2. Regime leakage: markets are non-stationary. A model trained on data from
   both 2019 and 2021 and tested on scattered days from the same two years
   isn't being asked "can you predict the *future* given only the *past*" —
   it's being asked "can you interpolate within a regime you've already seen
   examples from on both sides." That's not what happens in production,
   where you only ever have the past to predict the future.

Walk-forward validation fixes this by preserving time order: train on a
contiguous block, test on the contiguous block immediately after it, then
slide forward. This is the only validation scheme that matches how the
strategy would actually be deployed.

Why an embargo (purge gap) is still needed even with walk-forward
--------------------------------------------------------------------
Labels here are `horizon`-day forward returns (see labeling.py). Concretely,
with horizon=5: the label for the *last* day of the training window at date
T uses price data up to T+5. If the test window begins at T+1, its first
few labels use price information (T+1..T+5) that overlaps the exact dates
the last training label was computed from. The model didn't "see" test
features, but the training label distribution was constructed from price
moves that also generated the test set's earliest labels — a subtle but real
leak. The fix is to insert an embargo of `horizon` trading days between the
end of train and the start of test, so no label in train touches a date used
to construct a label in test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    train_dates: np.ndarray
    test_dates: np.ndarray
    fold_id: int


def walk_forward_splits(
    all_dates: np.ndarray,
    train_window_days: int,
    test_window_days: int,
    step_days: int,
    embargo_days: int,
) -> list[Fold]:
    """
    Generate expanding-then-sliding walk-forward folds over a sorted array of
    unique trading dates.

    Layout of one fold:
        [ ... train_window_days ... ] [ embargo_days ] [ ... test_window_days ... ]
        |------------------ moves forward by step_days each fold ---------------->

    Returns a list of Fold(train_dates, test_dates, fold_id).
    """
    all_dates = np.asarray(sorted(all_dates))
    n = len(all_dates)
    folds = []
    fold_id = 0

    train_start = 0
    while True:
        train_end = train_start + train_window_days
        test_start = train_end + embargo_days
        test_end = test_start + test_window_days

        if test_end > n:
            break

        train_dates = all_dates[train_start:train_end]
        test_dates = all_dates[test_start:test_end]

        folds.append(Fold(train_dates=train_dates, test_dates=test_dates, fold_id=fold_id))

        fold_id += 1
        train_start += step_days

    if not folds:
        raise ValueError(
            "No walk-forward folds could be created — the date range is shorter "
            "than train_window_days + embargo_days + test_window_days. "
            "Shorten the windows or widen the date range in config.yaml."
        )

    return folds
