"""
Factor correlation analysis.

Compute 6×6 cross-sectional factor correlation matrix monthly.
Flag pairs with |corr| > 0.6 as crowded / redundant.
"""

import numpy as np
import pandas as pd


def factor_correlation_matrix(
    factor_dict: dict,
    rebal_freq: str = "ME",
    crowding_threshold: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute average monthly cross-sectional factor correlation matrix.

    Parameters
    ----------
    factor_dict : dict
        Keys = factor names, values = signal DataFrames (dates × tickers).
    rebal_freq : str
        Pandas resample alias for rebalancing dates.
    crowding_threshold : float
        |corr| above this is flagged as crowded.

    Returns
    -------
    avg_corr : pd.DataFrame
        6×6 average correlation matrix.
    crowded_pairs : pd.DataFrame
        Pairs with |corr| > threshold.
    """
    factor_names = list(factor_dict.keys())
    # Use the first factor's index to determine dates
    base_index = list(factor_dict.values())[0].index
    rebal_dates = base_index.to_series().resample(rebal_freq).last().values

    monthly_corrs = []
    for d in rebal_dates:
        if d not in base_index:
            continue
        signals = {}
        for name, fdf in factor_dict.items():
            if d in fdf.index:
                signals[name] = fdf.loc[d]
        if len(signals) < 2:
            continue
        sig_df = pd.DataFrame(signals).dropna()
        if len(sig_df) < 20:
            continue
        corr = sig_df.corr(method="spearman")
        monthly_corrs.append(corr)

    if not monthly_corrs:
        empty = pd.DataFrame(index=factor_names, columns=factor_names, dtype=float)
        return empty, pd.DataFrame()

    avg_corr = pd.concat(monthly_corrs).groupby(level=0).mean()
    avg_corr = avg_corr.reindex(index=factor_names, columns=factor_names)

    # Identify crowded pairs
    crowded = []
    for i, f1 in enumerate(factor_names):
        for j, f2 in enumerate(factor_names):
            if j <= i:
                continue
            if f1 in avg_corr.index and f2 in avg_corr.columns:
                c = avg_corr.loc[f1, f2]
                if abs(c) > crowding_threshold:
                    crowded.append({"factor_1": f1, "factor_2": f2, "correlation": c})

    crowded_pairs = pd.DataFrame(crowded) if crowded else pd.DataFrame(
        columns=["factor_1", "factor_2", "correlation"]
    )
    return avg_corr, crowded_pairs
