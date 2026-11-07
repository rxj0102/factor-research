"""
Fama-MacBeth two-pass regression.

Step 1 (cross-sectional): For each month t, regress stock returns on
    standardized factor exposures. Store coefficients λ_t.
Step 2 (time-series): Average λ_t, compute Newey-West standard errors,
    derive t-statistics.
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac


def fama_macbeth_regression(
    factor_dict: dict,
    returns_1m: pd.DataFrame,
    newey_west_lags: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Fama-MacBeth cross-sectional regression.

    Parameters
    ----------
    factor_dict : dict
        Keys = factor names, values = signal DataFrames (dates × tickers).
    returns_1m : pd.DataFrame
        Monthly stock returns (dates × tickers).
    newey_west_lags : int
        Lags for Newey-West HAC covariance.

    Returns
    -------
    coef_ts : pd.DataFrame
        Time-series of cross-sectional coefficients (dates × factors).
    summary : pd.DataFrame
        mean_coef, t_stat, p_value, significant per factor.
    """
    factor_names = list(factor_dict.keys())

    # Get common rebalancing dates
    dates = returns_1m.index
    coef_records = []

    for d in dates:
        if d not in returns_1m.index:
            continue

        # Build cross-sectional dataset for date d
        rows = {}
        for name, fdf in factor_dict.items():
            if d not in fdf.index:
                continue
            rows[name] = fdf.loc[d]

        if len(rows) < len(factor_names):
            continue

        factor_df = pd.DataFrame(rows)
        ret = returns_1m.loc[d]

        # Common tickers
        common = factor_df.dropna().index.intersection(ret.dropna().index)
        if len(common) < 30:
            continue

        X = factor_df.loc[common]
        # Cross-sectional z-score
        X = (X - X.mean()) / X.std().replace(0, np.nan)
        X = X.dropna(axis=0)
        common = X.index

        y = ret[common]
        common = X.index.intersection(y.dropna().index)
        X = X.loc[common]
        y = y.loc[common]

        if len(common) < 20:
            continue

        try:
            model = OLS(y, add_constant(X)).fit()
            coefs = model.params.drop("const", errors="ignore")
            record = {"date": d}
            record.update(coefs.to_dict())
            coef_records.append(record)
        except Exception:
            continue

    coef_ts = pd.DataFrame(coef_records).set_index("date")

    # Newey-West t-stats
    summary_rows = []
    for factor in factor_names:
        if factor not in coef_ts.columns:
            continue
        series = coef_ts[factor].dropna()
        n = len(series)
        mean_c = series.mean()
        # Newey-West variance estimate
        lags = min(newey_west_lags, n - 1)
        gamma0 = np.var(series, ddof=1)
        nw_var = gamma0
        for lag in range(1, lags + 1):
            weight = 1 - lag / (lags + 1)
            gamma_l = np.mean(
                (series.iloc[lag:].values - mean_c) *
                (series.iloc[:-lag].values - mean_c)
            )
            nw_var += 2 * weight * gamma_l
        se = np.sqrt(max(nw_var, 1e-12) / n)
        t_stat = mean_c / se if se > 0 else np.nan
        from scipy.stats import t as t_dist
        p_val = 2 * t_dist.sf(abs(t_stat), df=n - 1) if not np.isnan(t_stat) else np.nan
        summary_rows.append(
            {
                "factor": factor,
                "mean_coef": mean_c,
                "t_stat": t_stat,
                "p_value": p_val,
                "significant": abs(t_stat) > 1.96 if not np.isnan(t_stat) else False,
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("factor")
    return coef_ts, summary
