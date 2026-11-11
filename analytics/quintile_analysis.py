"""
Quintile portfolio analysis.

Rank stocks into 5 equal groups each month based on factor signal.
Compute equal-weight average 20-day forward return per quintile.
Plot Q1–Q5 spread over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def quintile_portfolios(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    fwd_days: int = 20,
    n_quintiles: int = 5,
    rebal_freq: str = "ME",
) -> pd.DataFrame:
    """
    Compute monthly quintile portfolio returns.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor signal (dates × tickers).
    prices : pd.DataFrame
        Adjusted close prices (dates × tickers).
    fwd_days : int
        Forward return horizon in trading days.
    n_quintiles : int
        Number of quantile buckets (default 5).
    rebal_freq : str
        Pandas offset alias for rebalancing ('ME' = month-end).

    Returns
    -------
    pd.DataFrame
        Columns Q1…Q5 + spread (Q5 - Q1), monthly index.
    """
    fwd_ret = prices.shift(-fwd_days) / prices - 1

    # Month-end rebalancing dates
    # last trading day of each month; filtering calendar month-ends on the
    # index skipped months that end on a weekend or holiday
    if rebal_freq != "ME":
        raise ValueError("only month-end rebalancing is supported")
    idx = factor.index.to_series()
    rebal_dates = list(idx.groupby(factor.index.to_period("M")).max())

    rows = []
    for d in rebal_dates:
        sig = factor.loc[d].dropna()
        fwd = fwd_ret.loc[d].dropna()
        common = sig.index.intersection(fwd.index)
        if len(common) < n_quintiles * 5:
            continue
        sig_c = sig[common]
        fwd_c = fwd[common]
        labels = pd.qcut(sig_c, n_quintiles, labels=False, duplicates="drop")
        row = {"date": d}
        for q in range(n_quintiles):
            mask = labels == q
            row[f"Q{q + 1}"] = fwd_c[mask].mean()
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")
    df["spread"] = df[f"Q{n_quintiles}"] - df["Q1"]
    return df


def plot_quintile_spreads(
    quintile_dict: dict,
    save_path: str = None,
):
    """
    Plot 6-panel quintile return bar/line charts.

    Parameters
    ----------
    quintile_dict : dict
        Keys = factor names, values = DataFrames from quintile_portfolios().
    save_path : str
        If provided, save figure.
    """
    factors = list(quintile_dict.keys())
    n = len(factors)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()

    for i, factor in enumerate(factors):
        ax = axes[i]
        df = quintile_dict[factor]
        q_cols = [c for c in df.columns if c.startswith("Q")]
        means = df[q_cols].mean() * 100  # in percent

        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(q_cols)))
        bars = ax.bar(q_cols, means, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{factor} — Quintile 20d Fwd Returns", fontsize=11, fontweight="bold")
        ax.set_xlabel("Quintile (Q1=low signal, Q5=high signal)")
        ax.set_ylabel("Avg 20d Fwd Return (%)")
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate spread
        spread = means.iloc[-1] - means.iloc[0]
        ax.text(
            0.97, 0.97, f"Spread: {spread:.2f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="navy",
        )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Quintile Portfolio Analysis (20-Day Forward Returns)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
