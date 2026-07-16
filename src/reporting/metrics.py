"""
Risk-adjusted performance reporting — the numbers a PM actually asks for,
not raw accuracy. Every function takes a daily returns Series (net of
costs) and/or an equity curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(daily_returns: pd.Series) -> float:
    total_return = (1 + daily_returns).prod() - 1
    n_years = len(daily_returns) / TRADING_DAYS
    if n_years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / n_years) - 1


def annualized_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = daily_returns - rf_daily
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS)


def sortino_ratio(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def calmar_ratio(daily_returns: pd.Series, equity_curve: pd.Series) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return annualized_return(daily_returns) / mdd


def hit_rate(daily_returns: pd.Series) -> float:
    nonzero = daily_returns[daily_returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def avg_turnover(turnover: pd.Series) -> float:
    return float(turnover.mean())


def full_report(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    risk_free_annual: float = 0.04,
) -> dict:
    return {
        "annualized_return": annualized_return(daily_returns),
        "annualized_vol": annualized_vol(daily_returns),
        "sharpe_ratio": sharpe_ratio(daily_returns, risk_free_annual),
        "sortino_ratio": sortino_ratio(daily_returns, risk_free_annual),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(daily_returns, equity_curve),
        "hit_rate": hit_rate(daily_returns),
        "avg_daily_turnover": avg_turnover(turnover),
        "total_costs_paid": float(costs.sum()),
        "total_costs_as_pct_of_capital": float(costs.sum()),  # costs already in return units
        "final_equity": float(equity_curve.iloc[-1]) if len(equity_curve) else float("nan"),
    }
