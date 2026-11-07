"""
Value factor: Book-to-Market ratio.

Signal = book_value / market_cap. Higher B/M → cheaper → higher rank.
Fundamentals are updated quarterly; we forward-fill to daily frequency.
"""

import pandas as pd


def compute_value(
    market_cap: pd.DataFrame,
    book_value: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cross-sectional value signal (Book-to-Market).

    Parameters
    ----------
    market_cap : pd.DataFrame
        Daily market cap per ticker, shape (dates, tickers).
    book_value : pd.DataFrame
        Quarterly book value per ticker, reindexed to daily and forward-filled.

    Returns
    -------
    pd.DataFrame
        B/M ratio, same date index as market_cap.
    """
    # Align on common tickers and dates
    tickers = market_cap.columns.intersection(book_value.columns)
    bv = book_value[tickers].reindex(market_cap.index, method="ffill")
    mc = market_cap[tickers]

    bm = bv / mc.replace(0, float("nan"))
    return bm
