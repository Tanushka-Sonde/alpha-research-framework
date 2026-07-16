"""
LSTM sequence model — stretch variant.

Requires torch (pip install -r requirements-optional.txt). Imported lazily
so the rest of the framework works with zero torch dependency if you never
touch this.

Rationale for including it despite GBM being the default: it shows you can
model sequential dependence directly (rather than hand-crafting all lag
features), and it's a natural place to discuss the tradeoff in an
interview — LSTMs need much more data and are more prone to overfitting
noise on daily-bar tabular finance data than GBMs, which is exactly the
kind of "why I'd pick approach A over B for THIS problem" discussion
quant interviewers want to hear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LSTMConfig:
    sequence_length: int = 21
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    early_stopping_patience: int = 6


def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Sliding-window sequences: sample i uses features[i-seq_len+1 : i+1] to predict y[i].
    Purely causal — no future rows enter any given sequence."""
    n = len(X)
    xs, ys, idxs = [], [], []
    for i in range(seq_len - 1, n):
        xs.append(X[i - seq_len + 1 : i + 1])
        ys.append(y[i])
        idxs.append(i)
    if not xs:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,)), np.empty((0,), dtype=int)
    return np.stack(xs), np.array(ys), np.array(idxs)


class LSTMSignalModel:
    def __init__(self, cfg: LSTMConfig):
        self.cfg = cfg
        self.model = None
        self.mean_ = None
        self.std_ = None

    def _require_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for the LSTM model. Install with: "
                "pip install -r requirements-optional.txt"
            ) from e

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        self._require_torch()
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.feature_names_ = list(X_train.columns)
        Xtr = X_train.values.astype(np.float32)
        ytr = y_train.values.astype(np.float32)

        # Standardize using TRAIN stats only, applied to val too (no leakage)
        self.mean_ = Xtr.mean(axis=0)
        self.std_ = Xtr.std(axis=0) + 1e-8
        Xtr = (Xtr - self.mean_) / self.std_

        seq_len = self.cfg.sequence_length
        Xseq, yseq, _ = _build_sequences(Xtr, ytr, seq_len)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class LSTMNet(nn.Module):
            def __init__(self, n_features, hidden, layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features, hidden_size=hidden, num_layers=layers,
                    batch_first=True, dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last).squeeze(-1)

        self.model = LSTMNet(Xtr.shape[1], self.cfg.hidden_size, self.cfg.num_layers, self.cfg.dropout).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        loss_fn = nn.MSELoss()

        train_ds = TensorDataset(torch.tensor(Xseq), torch.tensor(yseq))
        train_dl = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)

        val_dl = None
        if X_val is not None and y_val is not None and len(X_val) > seq_len:
            Xv = X_val.values.astype(np.float32)
            Xv = (Xv - self.mean_) / self.std_
            yv = y_val.values.astype(np.float32)
            Xvseq, yvseq, _ = _build_sequences(Xv, yv, seq_len)
            if len(Xvseq) > 0:
                val_ds = TensorDataset(torch.tensor(Xvseq), torch.tensor(yvseq))
                val_dl = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)

        best_val = float("inf")
        patience_left = self.cfg.early_stopping_patience
        best_state = None

        for epoch in range(self.cfg.epochs):
            self.model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            if val_dl is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_dl:
                        xb, yb = xb.to(device), yb.to(device)
                        pred = self.model(xb)
                        val_losses.append(loss_fn(pred, yb).item())
                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                if val_loss < best_val:
                    best_val = val_loss
                    patience_left = self.cfg.early_stopping_patience
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._device = device
        self._seq_len = seq_len
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predictions aligned to X.index, with NaN for the first
        (seq_len - 1) rows that don't have enough history to form a sequence."""
        self._require_torch()
        import torch

        Xv = X.values.astype(np.float32)
        Xv = (Xv - self.mean_) / self.std_
        dummy_y = np.zeros(len(Xv), dtype=np.float32)
        Xseq, _, idxs = _build_sequences(Xv, dummy_y, self._seq_len)

        preds_full = np.full(len(X), np.nan)
        if len(Xseq) == 0:
            return preds_full

        self.model.eval()
        with torch.no_grad():
            batch = torch.tensor(Xseq).to(self._device)
            preds = self.model(batch).cpu().numpy()
        preds_full[idxs] = preds
        return preds_full
