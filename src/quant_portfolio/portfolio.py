from __future__ import annotations

import pandas as pd


def select_top_k(scores: pd.Series, top_k: int) -> list[str]:
    """Return the highest-scoring tickers."""
    clean_scores = scores.dropna().sort_values(ascending=False, kind="mergesort")
    return clean_scores.head(top_k).index.tolist()


def construct_weights(
    scores: pd.Series,
    top_k: int,
    volatility: pd.Series | None = None,
    weighting: str = "equal",
) -> pd.Series:
    """Construct long-only top-k weights."""
    selected = select_top_k(scores, top_k)
    weights = pd.Series(0.0, index=scores.index, dtype=float)
    if not selected:
        return weights

    if weighting == "equal" or volatility is None:
        weights.loc[selected] = 1.0 / len(selected)
    elif weighting == "inverse_vol":
        inv_vol = 1.0 / volatility.reindex(selected).replace(0.0, pd.NA).astype(float)
        inv_vol = inv_vol.replace([float("inf"), float("-inf")], pd.NA).dropna()
        if inv_vol.empty:
            weights.loc[selected] = 1.0 / len(selected)
        else:
            weights.loc[inv_vol.index] = inv_vol / inv_vol.sum()
    else:
        raise ValueError(f"Unsupported weighting method: {weighting}")

    return weights.fillna(0.0)


def compute_turnover(previous_weights: pd.Series, new_weights: pd.Series) -> float:
    """One-way portfolio turnover, based on absolute active weight change."""
    index = previous_weights.index.union(new_weights.index)
    previous = previous_weights.reindex(index, fill_value=0.0)
    new = new_weights.reindex(index, fill_value=0.0)
    return float((new - previous).abs().sum() / 2.0)
