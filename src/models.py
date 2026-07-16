"""
Signal generation models.

Gradient boosting (LightGBM) is the default and the one actually used by the
backtest: it's what most quant shops use for tabular alpha signals in
practice — it handles nonlinear interactions and non-stationary feature
scales far better than linear models, trains in seconds, and is easy to
explain via feature importance.

An LSTM variant is included as a stretch goal to demonstrate range on
sequence models, but it's NOT required to run the core pipeline — it's only
invoked if config.yaml sets model.type: "lstm", and it degrades gracefully
(raises a clear error) if torch isn't installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


class GradientBoostingSignal:
    """Thin wrapper around LightGBM for cross-sectional return prediction."""

    def __init__(self, params: dict):
        self.params = params
        self.model = LGBMRegressor(**params, verbosity=-1)
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingSignal":
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names_])

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names_
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Stretch goal: LSTM sequence model over a rolling feature window per ticker.
# Only imported/instantiated if explicitly requested — keeps torch optional.
# ---------------------------------------------------------------------------
class LSTMSignal:
    """
    Small LSTM over a trailing window of features per ticker, predicting the
    same forward-return label as the gradient-boosting model. Included to
    show range beyond tabular GBMs; not the recommended default because it
    needs far more data per ticker to avoid overfitting and is much slower
    to iterate on than LightGBM during research.
    """

    def __init__(self, n_features: int, seq_len: int = 20, hidden_size: int = 32, epochs: int = 15, lr: float = 1e-3):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "LSTMSignal requires torch. Install it with: "
                "pip install -r requirements-optional.txt"
            ) from e

        self._torch = torch
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.net = _Net()

    def _make_sequences(self, X: pd.DataFrame, y: pd.Series | None, ticker_col: pd.Series):
        """Build (n_samples, seq_len, n_features) tensors per ticker, respecting
        time order so no sequence crosses ticker boundaries or looks ahead."""
        torch = self._torch
        xs, ys = [], []
        for ticker, idx in ticker_col.groupby(ticker_col).groups.items():
            xt = X.loc[idx].values
            if len(xt) <= self.seq_len:
                continue
            for i in range(self.seq_len, len(xt)):
                xs.append(xt[i - self.seq_len : i])
                if y is not None:
                    ys.append(y.loc[idx].values[i])
        X_seq = torch.tensor(np.array(xs), dtype=torch.float32)
        y_seq = torch.tensor(np.array(ys), dtype=torch.float32) if y is not None else None
        return X_seq, y_seq

    def fit(self, X: pd.DataFrame, y: pd.Series, ticker_col: pd.Series) -> "LSTMSignal":
        torch = self._torch
        X_seq, y_seq = self._make_sequences(X, y, ticker_col)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.net(X_seq)
            loss = loss_fn(pred, y_seq)
            loss.backward()
            opt.step()
        return self

    def predict(self, X: pd.DataFrame, ticker_col: pd.Series) -> np.ndarray:
        torch = self._torch
        self.net.eval()
        X_seq, _ = self._make_sequences(X, None, ticker_col)
        with torch.no_grad():
            return self.net(X_seq).numpy()
