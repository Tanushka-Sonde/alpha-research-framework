"""
Risk-adjusted performance reporting — the metrics a PM actually asks about,
not "accuracy". Accuracy on a forward-return sign is nearly meaningless
without knowing the P&L, cost, and drawdown profile that comes with it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(daily_returns: pd.Series) -> float:
    total_growth = (1 + daily_returns).prod()
    n_years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return np.nan
    return total_growth ** (1 / n_years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min()


def calmar_ratio(daily_returns: pd.Series, equity_curve: pd.Series) -> float:
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return np.nan
    return annualized_return(daily_returns) / abs(mdd)


def hit_rate(daily_returns: pd.Series) -> float:
    """Fraction of days with positive net strategy return."""
    nonzero = daily_returns[daily_returns != 0]
    if len(nonzero) == 0:
        return np.nan
    return (nonzero > 0).mean()


def average_turnover(turnover: pd.Series) -> float:
    return turnover.mean()


def summarize(daily_returns: pd.Series, equity_curve: pd.Series, turnover: pd.Series) -> dict:
    """One-stop shop: everything the dashboard and README table need."""
    return {
        "annualized_return": annualized_return(daily_returns),
        "annualized_volatility": annualized_volatility(daily_returns),
        "sharpe_ratio": sharpe_ratio(daily_returns),
        "sortino_ratio": sortino_ratio(daily_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(daily_returns, equity_curve),
        "hit_rate": hit_rate(daily_returns),
        "avg_daily_turnover": average_turnover(turnover),
        "total_return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0,
    }
