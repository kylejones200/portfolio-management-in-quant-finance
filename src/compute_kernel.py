"""Portfolio mean and volatility from weights and row-major returns."""

from __future__ import annotations

import numpy as np


def portfolio_stats(
    weights: np.ndarray,
    returns: np.ndarray,
    n_assets: int,
    n_periods: int,
) -> tuple[float, float]:
    w = np.asarray(weights, dtype=float)
    r = np.asarray(returns, dtype=float).reshape(n_periods, n_assets)
    port = r @ w
    mean = float(port.mean())
    var = float(((port - mean) ** 2).sum() / n_periods)
    return mean, var**0.5
