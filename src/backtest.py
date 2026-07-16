"""
Backtest engine — simulates actually trading the signal, not just checking
whether its sign matched the realized return. That distinction is the whole
point of a credible backtest:

- Position sizing: cross-sectional rank into long/short quantile buckets,
  equal-weighted within each bucket, capped per-name at max_position_pct of
  gross capital (no single name can blow up the book).
- Transaction costs: applied in bps on every dollar of *traded* notional
  (i.e. on position changes, not on the full position each day).
- Slippage: modeled as an additional bps cost on traded notional, applied
  symmetrically — a simple but standard way to approximate market impact
  without needing L2 order book data.
- Turnover is tracked explicitly since it's what eats the edge in practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame          # date x ticker, weight of gross capital
    turnover: pd.Series              # daily turnover as fraction of gross capital
    trades_cost: pd.Series           # daily cost drag (transaction cost + slippage)
    gross_returns: pd.Series         # daily strategy return before costs


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float,
        long_quantile: float,
        short_quantile: float,
        max_position_pct: float,
        transaction_cost_bps: float,
        slippage_bps: float,
    ):
        self.initial_capital = initial_capital
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile
        self.max_position_pct = max_position_pct
        self.cost_rate = (transaction_cost_bps + slippage_bps) / 10_000.0

    def _target_weights(self, signal_row: pd.Series) -> pd.Series:
        """
        Turn one day's cross-sectional signal into target portfolio weights:
        equal-weight long the top quantile, equal-weight short the bottom
        quantile, capped per name, dollar-neutral if both sides are active.
        """
        s = signal_row.dropna()
        if s.empty:
            return pd.Series(dtype=float)

        n = len(s)
        n_long = max(1, int(np.floor(n * self.long_quantile))) if self.long_quantile > 0 else 0
        n_short = max(1, int(np.floor(n * self.short_quantile))) if self.short_quantile > 0 else 0

        ranked = s.sort_values(ascending=False)
        longs = ranked.index[:n_long] if n_long > 0 else []
        shorts = ranked.index[-n_short:] if n_short > 0 else []

        weights = pd.Series(0.0, index=s.index)
        if len(longs) > 0:
            w = min(1.0 / len(longs), self.max_position_pct)
            weights.loc[longs] = w
        if len(shorts) > 0:
            w = min(1.0 / len(shorts), self.max_position_pct)
            weights.loc[shorts] = -w

        # Normalize gross exposure to 1.0 (100% of capital deployed gross)
        gross = weights.abs().sum()
        if gross > 0:
            weights = weights / gross
        return weights

    def run(self, signals: pd.DataFrame, forward_returns: pd.DataFrame) -> BacktestResult:
        """
        signals: date x ticker matrix of model scores (higher = more bullish),
                 already out-of-sample (from the walk-forward loop).
        forward_returns: date x ticker matrix of the *realized* 1-day-forward
                 return for that ticker, used to mark the position to market.
                 Must be shifted so that a position formed using signal at
                 date t earns the return realized from t to t+1 (no
                 lookahead — see scripts/run_pipeline.py for how this is built).
        """
        dates = signals.index
        tickers = signals.columns

        weights_history = pd.DataFrame(0.0, index=dates, columns=tickers)
        prev_weights = pd.Series(0.0, index=tickers)

        gross_returns = pd.Series(0.0, index=dates)
        turnover = pd.Series(0.0, index=dates)
        trades_cost = pd.Series(0.0, index=dates)

        for date in dates:
            target = self._target_weights(signals.loc[date])
            target = target.reindex(tickers).fillna(0.0)

            traded_notional = (target - prev_weights).abs().sum()
            turnover.loc[date] = traded_notional
            trades_cost.loc[date] = traded_notional * self.cost_rate

            day_fwd_ret = forward_returns.loc[date].reindex(tickers).fillna(0.0)
            gross_returns.loc[date] = (target * day_fwd_ret).sum()

            weights_history.loc[date] = target
            prev_weights = target

        net_returns = gross_returns - trades_cost
        equity_curve = self.initial_capital * (1 + net_returns).cumprod()

        return BacktestResult(
            equity_curve=equity_curve,
            daily_returns=net_returns,
            positions=weights_history,
            turnover=turnover,
            trades_cost=trades_cost,
            gross_returns=gross_returns,
        )
