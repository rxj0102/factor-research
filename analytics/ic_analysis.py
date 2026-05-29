"""
Rolling Information Coefficient (IC) and IC Information Ratio (ICIR).

IC = Spearman rank correlation between factor signal and forward return.
ICIR = mean(IC) / std(IC) over a rolling window.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _cross_sectional_ic(signal_row: pd.Series, fwd_row: pd.Series) -> float:
    """Spearman IC for a single cross-section."""
    common = signal_row.index.intersection(fwd_row.index)
    s = signal_row[common].dropna()
    f = fwd_row[common].dropna()
    common2 = s.index.intersection(f.index)
    if len(common2) < 10:
        return float("nan")
    rho, _ = stats.spearmanr(s[common2], f[common2])
    return rho


def rolling_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    window: int = 252,
) -> pd.Series:
    """
    Compute rolling cross-sectional Spearman IC.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor signal, shape (dates, tickers).
    forward_returns : pd.DataFrame
        Forward return (e.g., 1-day, 5-day, 20-day), same shape.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        Rolling mean IC indexed by date.
    """
    # Compute daily IC first
    dates = factor.index.intersection(forward_returns.index)
    daily_ic = pd.Series(index=dates, dtype=float)
    for d in dates:
        daily_ic[d] = _cross_sectional_ic(factor.loc[d], forward_returns.loc[d])

    rolling_mean_ic = daily_ic.rolling(window, min_periods=max(20, window // 4)).mean()
    return rolling_mean_ic


def ic_summary(
    factor: pd.DataFrame,
    forward_returns_dict: dict,
    window: int = 252,
) -> pd.DataFrame:
    """
    Summarise IC and ICIR across multiple forward-return horizons.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor signal.
    forward_returns_dict : dict
        Keys are horizon labels (e.g. '1d', '5d', '20d'),
        values are forward-return DataFrames.
    window : int
        Rolling window for ICIR computation.

    Returns
    -------
    pd.DataFrame
        Columns: horizon, mean_IC, std_IC, ICIR, IC_positive_pct.
    """
    rows = []
    dates = factor.index
    for label, fwd in forward_returns_dict.items():
        daily_ic = pd.Series(dtype=float)
        common_dates = factor.index.intersection(fwd.index)
        ics = []
        for d in common_dates:
            ic_val = _cross_sectional_ic(factor.loc[d], fwd.loc[d])
            ics.append(ic_val)
        ic_series = pd.Series(ics, index=common_dates).dropna()
        if len(ic_series) == 0:
            continue
        rows.append(
            {
                "horizon": label,
                "mean_IC": ic_series.mean(),
                "std_IC": ic_series.std(),
                "ICIR": ic_series.mean() / ic_series.std() if ic_series.std() > 0 else np.nan,
                "IC_positive_pct": (ic_series > 0).mean() * 100,
            }
        )
    return pd.DataFrame(rows)
