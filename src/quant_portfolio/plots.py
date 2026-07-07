from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quant_portfolio_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity_curves(equity: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    equity.plot(ax=ax, linewidth=1.5)
    ax.set_title("Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, path)


def plot_drawdowns(equity: pd.DataFrame, path: str | Path) -> None:
    drawdown = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 6))
    drawdown.plot(ax=ax, linewidth=1.2)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, path)


def plot_rolling_sharpe(returns: pd.DataFrame, path: str | Path, window: int = 252) -> None:
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_vol.replace(0.0, np.nan)
    fig, ax = plt.subplots(figsize=(11, 6))
    rolling_sharpe.plot(ax=ax, linewidth=1.2)
    ax.set_title("Rolling 12-Month Sharpe")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, path)


def plot_allocation_heatmap(weights: pd.DataFrame, path: str | Path, strategy: str | None = None) -> None:
    frame = weights.copy()
    if {"strategy", "date", "ticker", "weight"}.issubset(frame.columns):
        if strategy is None:
            candidates = [value for value in frame["strategy"].unique() if "Momentum" in value]
            strategy = candidates[0] if candidates else frame["strategy"].iloc[0]
        frame = frame[frame["strategy"] == strategy]
        matrix = frame.pivot(index="date", columns="ticker", values="weight").fillna(0.0)
    else:
        matrix = frame.fillna(0.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"Allocation Heatmap: {strategy or 'Strategy'}")
    ax.set_yticks(range(len(matrix.columns)))
    ax.set_yticklabels(matrix.columns)
    if len(matrix.index) > 0:
        tick_positions = np.linspace(0, len(matrix.index) - 1, min(8, len(matrix.index)), dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([pd.Timestamp(matrix.index[i]).strftime("%Y-%m") for i in tick_positions], rotation=45)
    fig.colorbar(image, ax=ax, label="Weight")
    _save(fig, path)


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
