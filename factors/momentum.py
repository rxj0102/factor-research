"""
Momentum factor: Jegadeesh-Titman 12-1 month return.

Signal = cumulative return from t-252 to t-21 (skip most recent month).
Higher signal → higher rank (long winners, short losers).
"""

import pandas as pd


def compute_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional momentum signal.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, shape (dates, tickers).

    Returns
    -------
    pd.DataFrame
        Momentum signal, same shape as prices. NaN where insufficient history.
    """
    # 12-month lookback = 252 trading days; skip last month = 21 trading days
    lookback = 252
    skip = 21

    # Cumulative return from t-252 to t-21
    ret_12m = prices.shift(skip) / prices.shift(lookback) - 1
    return ret_12m
