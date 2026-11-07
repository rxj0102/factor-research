"""
Size factor: log market capitalization.

Signal = -log(market_cap). Negative sign so that smaller stocks → higher rank.
"""

import numpy as np
import pandas as pd


def compute_size(market_cap: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional size signal.

    Parameters
    ----------
    market_cap : pd.DataFrame
        Daily market cap per ticker in USD, shape (dates, tickers).

    Returns
    -------
    pd.DataFrame
        Negative log market cap, same shape as market_cap.
    """
    log_mc = np.log(market_cap.replace(0, float("nan")))
    return -log_mc  # negate: smaller = higher rank
