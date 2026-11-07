"""
Short-term Reversal factor: prior 1-week (5-day) return.

Signal = -ret_5d. Negative sign so that recent losers → higher rank (mean reversion).
"""

import pandas as pd


def compute_reversal(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute cross-sectional short-term reversal signal.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, shape (dates, tickers).
    window : int
        Lookback in trading days (default 5 ≈ 1 week).

    Returns
    -------
    pd.DataFrame
        Negative 5-day return, same shape as prices.
    """
    ret = prices / prices.shift(window) - 1
    return -ret  # negate: recent losers get positive (buy) signal
