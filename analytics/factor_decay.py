"""
Factor decay analysis: IC at multiple forward lags.

For each lag L in [1, 2, 3, 5, 10, 20] compute the full-sample mean IC.
Plots one decay curve per factor.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


LAGS = [1, 2, 3, 5, 10, 20]


def _ic_at_lag(factor: pd.DataFrame, prices: pd.DataFrame, lag: int) -> float:
    """Mean cross-sectional Spearman IC for a given forward lag."""
    fwd_ret = prices.shift(-lag) / prices - 1
    ics = []
    for d in factor.index:
        if d not in fwd_ret.index:
            continue
        s = factor.loc[d].dropna()
        f = fwd_ret.loc[d].dropna()
        common = s.index.intersection(f.index)
        if len(common) < 10:
            continue
        rho, _ = stats.spearmanr(s[common], f[common])
        ics.append(rho)
    return float(np.nanmean(ics)) if ics else float("nan")


def factor_decay_curve(
    factors: dict,
    prices: pd.DataFrame,
    lags: list = None,
) -> pd.DataFrame:
    """
    Compute IC decay for multiple factors.

    Parameters
    ----------
    factors : dict
        Keys are factor names, values are signal DataFrames.
    prices : pd.DataFrame
        Adjusted close prices used to compute forward returns.
    lags : list
        Forward lags in trading days.

    Returns
    -------
    pd.DataFrame
        Index = lags, columns = factor names, values = mean IC.
    """
    if lags is None:
        lags = LAGS
    results = {}
    for name, factor in factors.items():
        decay = {}
        for lag in lags:
            decay[lag] = _ic_at_lag(factor, prices, lag)
        results[name] = decay
    return pd.DataFrame(results, index=lags)


def plot_factor_decay(decay_df: pd.DataFrame, save_path: str = None):
    """
    Plot 6-panel factor decay curves.

    Parameters
    ----------
    decay_df : pd.DataFrame
        Output of factor_decay_curve().
    save_path : str
        If provided, save figure to this path.
    """
    factors = decay_df.columns.tolist()
    n = len(factors)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, factor in enumerate(factors):
        ax = axes[i]
        y = decay_df[factor]
        ax.plot(decay_df.index, y, marker="o", linewidth=2, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{factor} — IC Decay", fontsize=12, fontweight="bold")
        ax.set_xlabel("Forward Lag (days)")
        ax.set_ylabel("Mean IC (Spearman)")
        ax.set_xticks(decay_df.index)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Factor IC Decay Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
