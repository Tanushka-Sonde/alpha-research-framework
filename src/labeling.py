"""
Labeling — defines what the model is trying to predict.

label(t) = (close[t+horizon] / close[t]) - 1

This is *by construction* a forward-looking value, which is fine — it's the
label, not a feature. The danger is elsewhere:

1. The last `horizon` rows of each ticker's history have no valid label
   (there's no t+horizon yet) and must be dropped, not filled.
2. When a label uses `horizon` days of future information, any training
   fold whose *last* rows are within `horizon` days of a test fold's *first*
   row has a subtle leak: the label computed at the end of train touches
   dates that also appear at the start of test. This is why validation.py
   enforces an embargo period equal to `horizon` between train and test —
   see that module's docstring for the concrete example.
"""
from __future__ import annotations

import pandas as pd


def add_forward_return_label(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Adds a `label` column (forward `horizon`-day return) to a (date, ticker)
    panel. Rows where the label can't be computed (end of each ticker's
    series) are kept but will contain NaN — callers must drop NaNs before
    training, not fill them.
    """
    panel = panel.copy()
    labels = (
        panel.groupby(level="ticker", group_keys=False)["_close"]
        .apply(lambda s: s.shift(-horizon) / s - 1.0)
    )
    panel["label"] = labels
    return panel


def drop_unlabeled(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without a valid forward-return label (end-of-series tail)."""
    return panel.dropna(subset=["label"])
