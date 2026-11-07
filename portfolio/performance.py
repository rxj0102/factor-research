"""
Portfolio performance metrics.

Computes: annualized return, Sharpe ratio, max drawdown, turnover, DSR.
"""

import numpy as np
import pandas as pd

from analytics.deflated_sharpe import deflated_sharpe_ratio


def portfolio_performance(
    portfolio_df: pd.DataFrame,
    ret_col: str = "net_ret",
    periods_per_year: int = 252,
    n_trials: int = 6,
) -> dict:
    """
    Compute full performance metrics for a long-short portfolio.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Output of build_long_short_portfolio(); must contain ret_col.
    ret_col : str
        Column with daily returns to evaluate.
    periods_per_year : int
        252 for daily data.
    n_trials : int
        Number of strategies tested (for DSR multiple-testing correction).

    Returns
    -------
    dict of performance metrics.
    """
    rets = portfolio_df[ret_col].dropna()
    if len(rets) == 0:
        return {}

    cum = (1 + rets).cumprod()
    n_years = len(rets) / periods_per_year

    ann_ret = cum.iloc[-1] ** (1 / n_years) - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()

    # Annualized turnover
    turnover_col = "turnover"
    if turnover_col in portfolio_df.columns:
        avg_monthly_to = portfolio_df[turnover_col].sum() / max(n_years, 1)
    else:
        avg_monthly_to = np.nan

    # DSR
    dsr_result = deflated_sharpe_ratio(
        returns=rets.values,
        n_trials=n_trials,
        periods_per_year=periods_per_year,
    )

    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "annual_turnover": avg_monthly_to,
        "deflated_sharpe_ratio": dsr_result["dsr"],
        "dsr_p_value": dsr_result["p_value"],
        "n_days": len(rets),
        "n_years": n_years,
    }
