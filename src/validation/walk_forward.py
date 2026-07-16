"""
Purged walk-forward validation
===============================
Why not a random train/test split?

Because the label at time t (`fwd_ret_Hd`) is computed using price data
from t all the way to t+H. If you random-split rows into train/test, some
training rows will have labels whose underlying return window OVERLAPS
the test period (and vice versa) — the model ends up training on
information that is, chronologically, "from the future" relative to
some of its test points. That's lookahead bias hiding inside a metric
that otherwise looks perfectly legitimate (in-sample-style random CV can
show a great Sharpe that evaporates the moment you trade it live).

This module instead does:
1. WALK-FORWARD splits: train only on data strictly before the test
   fold (expanding window), never on future data relative to the test
   fold. This alone respects time ordering.
2. PURGING: any training sample whose label window [t, t+H] overlaps the
   test fold's date range is dropped from the training set. Respects
   time ordering isn't enough on its own if labels span multiple days.
3. EMBARGO: after the test fold, we skip an additional `embargo_days`
   before the *next* fold's training data can resume, because test-fold
   information (via serial correlation in returns/volatility) can leak
   backward into training-adjacent rows.

Reference for the general approach: Marcos Lopez de Prado,
"Advances in Financial Machine Learning" (2018), ch. 7 (Cross-Validation
in Finance) — purged K-Fold CV with embargo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    n_splits: int = 6
    min_train_size: int = 504
    test_size: int = 63
    embargo_days: int = 5
    label_horizon: int = 5  # used as the purge window


class PurgedWalkForward:
    """
    Iterates expanding-window train/test folds over a DatetimeIndex,
    purging label-overlap rows from train and applying a post-test embargo.
    """

    def __init__(self, cfg: WalkForwardConfig):
        self.cfg = cfg

    def split(self, index: pd.DatetimeIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(index)
        cfg = self.cfg
        positions = np.arange(n)

        # Figure out where test folds can start given min_train_size and n_splits
        first_test_start = cfg.min_train_size
        remaining = n - first_test_start
        if remaining < cfg.test_size:
            raise ValueError(
                f"Not enough data for even one fold: n={n}, "
                f"min_train_size={cfg.min_train_size}, test_size={cfg.test_size}"
            )

        # Evenly space up to n_splits fold-starts across the remaining data
        max_possible_splits = max(1, remaining // cfg.test_size)
        n_splits = min(cfg.n_splits, max_possible_splits)
        test_starts = [first_test_start + i * cfg.test_size for i in range(n_splits)]
        test_starts = [s for s in test_starts if s + cfg.test_size <= n]

        for test_start in test_starts:
            test_end = min(test_start + cfg.test_size, n)  # exclusive
            test_idx = positions[test_start:test_end]

            # --- Purge: drop training rows whose label window overlaps the test fold ---
            # A row at position p has a label spanning [p, p + label_horizon].
            # It must be dropped from train if that span touches [test_start, test_end).
            purge_start = max(0, test_start - cfg.label_horizon)
            train_candidate_end = purge_start  # train only goes up to here (exclusive)

            # --- Embargo applies to the *next* fold's usable training window ---
            # (handled naturally since each fold's train_end is this fold's purge_start;
            #  we additionally push the embargo forward so overlap after the test fold
            #  is excluded from ever entering training in a later fold)
            train_idx = positions[0:train_candidate_end]

            # Remove the embargo zone that trails this test fold from any FUTURE
            # fold's training set by excluding [test_end, test_end+embargo) from
            # train_idx of subsequent folds — implemented by tracking a global
            # exclusion mask across folds.
            train_idx = self._apply_prior_embargoes(train_idx, test_starts, test_start, test_end, n)

            if len(train_idx) < cfg.min_train_size // 2:
                continue  # skip degenerate folds

            yield train_idx, test_idx

    def _apply_prior_embargoes(self, train_idx, all_test_starts, current_test_start, current_test_end, n):
        """Excludes embargo zones of all PRIOR test folds from this fold's train set."""
        cfg = self.cfg
        mask = np.ones(len(train_idx), dtype=bool)
        idx_positions = train_idx
        for ts in all_test_starts:
            if ts >= current_test_start:
                continue
            te = min(ts + cfg.test_size, n)
            embargo_end = min(te + cfg.embargo_days, n)
            # exclude any train position inside [te, embargo_end)
            in_embargo = (idx_positions >= te) & (idx_positions < embargo_end)
            mask &= ~in_embargo
        return idx_positions[mask]

    def n_folds(self, index: pd.DatetimeIndex) -> int:
        return sum(1 for _ in self.split(index))
