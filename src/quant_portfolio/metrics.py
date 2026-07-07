from __future__ import annotations

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 252


def cumulative_return(returns: pd.Series) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    ending = (1.0 + returns).prod()
    if ending <= 0:
        return -1.0
    return float(ending ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    returns = _clean_returns(returns)
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    returns = _clean_returns(returns)
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0.0:
        return 0.0
    return float(returns.mean() * periods_per_year / vol)


def sortino_ratio(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    returns = _clean_returns(returns)
    downside = returns[returns < 0.0]
    if len(downside) < 2:
        return 0.0
    downside_vol = downside.std(ddof=1) * np.sqrt(periods_per_year)
    if downside_vol == 0.0:
        return 0.0
    return float(returns.mean() * periods_per_year / downside_vol)


def max_drawdown(returns: pd.Series) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0.0:
        return 0.0
    return float(cagr(returns) / mdd)


def hit_rate(returns: pd.Series) -> float:
    returns = _clean_returns(returns)
    if returns.empty:
        return 0.0
    return float((returns > 0.0).mean())


def summarize_returns(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    initial_turnover: float = 0.0,
) -> dict[str, float]:
    returns = _clean_returns(returns)
    final_equity = float((1.0 + returns).prod()) if not returns.empty else 1.0
    average_turnover = 0.0 if turnover is None or turnover.empty else float(turnover.mean())
    return {
        "cumulative_return": cumulative_return(returns),
        "cagr": cagr(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
        "hit_rate": hit_rate(returns),
        "average_turnover": average_turnover,
        "initial_turnover": float(initial_turnover),
        "final_equity": final_equity,
    }


def _clean_returns(returns: pd.Series) -> pd.Series:
    if returns is None:
        return pd.Series(dtype=float)
    return pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
