from __future__ import annotations

import pandas as pd

from quant_portfolio.metrics import max_drawdown, sharpe_ratio


def test_max_drawdown_calculation() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])
    assert round(max_drawdown(returns), 6) == -0.2


def test_sharpe_does_not_crash_on_zero_volatility() -> None:
    returns = pd.Series([0.0, 0.0, 0.0])
    assert sharpe_ratio(returns) == 0.0
