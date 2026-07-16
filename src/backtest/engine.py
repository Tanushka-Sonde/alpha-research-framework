"""
Backtest engine
================
Converts a cross-sectional (or single-asset) predicted-return signal into
actual traded positions, then simulates a portfolio with realistic
frictions. This is deliberately NOT "did the sign match" — it charges you
for every rebalance and scales positions like a real book would.

Pipeline inside `run()`:
  1. signal -> target weights (quantile long/short, or sign, or
     zscore-scaled), cross-sectionally normalized per rebalance date
  2. volatility targeting: scale gross exposure so realized portfolio
     vol tracks a target annual vol (common PM-level risk control)
  3. position caps: no single name allowed to dominate the book
  4. turnover-based transaction costs + slippage charged on every
     change in position, not just at entry
  5. daily P&L accounting -> equity curve
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000
    transaction_cost_bps: float = 5
    slippage_bps: float = 2
    signal_to_position: str = "quantile"  # "quantile" | "sign" | "zscore_scaled"
    long_quantile: float = 0.8
    short_quantile: float = 0.2
    allow_short: bool = True
    vol_target_annual: float = 0.15
    vol_lookback: int = 21
    max_position_weight: float = 0.35
    rebalance_frequency: str = "daily"


class BacktestEngine:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def _signal_to_raw_weights(self, signal_row: pd.Series) -> pd.Series:
        """Cross-sectional conversion of one date's predicted returns into
        raw (pre-vol-scaling) target weights that sum |w| to <= 1 gross."""
        cfg = self.cfg
        s = signal_row.dropna()
        if s.empty:
            return pd.Series(dtype=float)

        if cfg.signal_to_position == "sign":
            w = np.sign(s)
            if not cfg.allow_short:
                w = w.clip(lower=0)

        elif cfg.signal_to_position == "zscore_scaled":
            z = (s - s.mean()) / (s.std() + 1e-12)
            w = z.clip(-3, 3)
            if not cfg.allow_short:
                w = w.clip(lower=0)

        elif cfg.signal_to_position == "quantile":
            long_cut = s.quantile(cfg.long_quantile)
            short_cut = s.quantile(cfg.short_quantile)
            w = pd.Series(0.0, index=s.index)
            w[s >= long_cut] = 1.0
            if cfg.allow_short:
                w[s <= short_cut] = -1.0
        else:
            raise ValueError(f"Unknown signal_to_position: {cfg.signal_to_position}")

        gross = w.abs().sum()
        if gross > 0:
            w = w / gross  # normalize so gross exposure == 1 before vol targeting/caps
        # position cap, renormalize after capping so gross stays sane
        w = w.clip(-cfg.max_position_weight, cfg.max_position_weight)
        return w

    def run(self, signals: pd.DataFrame, forward_returns_realized: pd.DataFrame) -> dict:
        """
        signals: DataFrame [date x ticker] of predicted forward returns (model output),
                 indexed by the date the prediction was MADE (i.e. decision date).
        forward_returns_realized: DataFrame [date x ticker] of the ACTUAL 1-day return
                 realized on each date (used to compute P&L day by day, not the
                 forward-horizon return — daily rebalancing needs daily realized returns).

        Returns dict with equity_curve, daily_returns, weights_history, turnover, costs.
        """
        cfg = self.cfg
        dates = signals.index
        tickers = signals.columns

        weights_history = pd.DataFrame(0.0, index=dates, columns=tickers)
        raw_weights_prev = pd.Series(0.0, index=tickers)

        daily_port_returns = pd.Series(0.0, index=dates)
        turnover_series = pd.Series(0.0, index=dates)
        cost_series = pd.Series(0.0, index=dates)

        realized_vol = 1.0  # running estimate, seeded neutral

        recent_port_rets = []

        for i, date in enumerate(dates):
            raw_w = self._signal_to_raw_weights(signals.loc[date])
            raw_w = raw_w.reindex(tickers).fillna(0.0)

            # --- Volatility targeting: scale gross exposure using trailing realized vol ---
            if len(recent_port_rets) >= max(5, cfg.vol_lookback // 2):
                trailing = np.array(recent_port_rets[-cfg.vol_lookback:])
                realized_vol = trailing.std() * np.sqrt(252)
                realized_vol = max(realized_vol, 1e-4)
            vol_scale = cfg.vol_target_annual / realized_vol
            vol_scale = float(np.clip(vol_scale, 0.1, 5.0))  # avoid absurd leverage swings

            target_w = (raw_w * vol_scale).clip(-cfg.max_position_weight, cfg.max_position_weight)
            weights_history.loc[date] = target_w

            # --- Turnover & costs (charged on the CHANGE in weight, applied same day) ---
            turnover = (target_w - raw_weights_prev).abs().sum()
            turnover_series.loc[date] = turnover
            cost_bps = cfg.transaction_cost_bps + cfg.slippage_bps
            cost = turnover * (cost_bps / 10_000.0)
            cost_series.loc[date] = cost

            # --- P&L: today's realized returns applied to YESTERDAY's positions ---
            # (you can't earn today's return on a position you only just took today
            #  intraday close-to-close accounting — position set at close t-1, earns
            #  return from t-1 to t)
            todays_realized = forward_returns_realized.loc[date].reindex(tickers).fillna(0.0)
            gross_pnl = (raw_weights_prev * todays_realized).sum()
            net_pnl = gross_pnl - cost
            daily_port_returns.loc[date] = net_pnl
            recent_port_rets.append(net_pnl)

            raw_weights_prev = target_w

        equity_curve = cfg.initial_capital * (1 + daily_port_returns).cumprod()

        return {
            "equity_curve": equity_curve,
            "daily_returns": daily_port_returns,
            "weights_history": weights_history,
            "turnover": turnover_series,
            "costs": cost_series,
        }
