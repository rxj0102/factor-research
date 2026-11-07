"""
Low Volatility factor: 20-day realized volatility.

Signal = -stddev(daily_returns, 20). Negative sign so that lower vol → higher rank.
"""

import numpy as np
import pandas as pd


def compute_low_volatility(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute cross-sectional low-volatility signal.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, shape (dates, tickers).
    window : int
        Rolling window for realized volatility in trading days.

    Returns
    -------
    pd.DataFrame
        Negative realized volatility (annualized), same shape as prices.
    """
    daily_returns = prices.pct_change()
    vol = daily_returns.rolling(window).std() * np.sqrt(252)
    return -vol  # negate so smaller vol → larger signal
