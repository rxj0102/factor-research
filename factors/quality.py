"""
Quality factor: Return on Equity (ROE).

Signal = net_income / book_equity. Higher ROE → higher quality → higher rank.
Fundamentals are updated quarterly; we forward-fill to daily frequency.
"""

import pandas as pd


def compute_quality(
    net_income: pd.DataFrame,
    book_equity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cross-sectional quality signal (ROE).

    Parameters
    ----------
    net_income : pd.DataFrame
        Trailing twelve-month net income, quarterly, indexed by date and ticker.
    book_equity : pd.DataFrame
        Book equity (shareholders' equity), quarterly.

    Returns
    -------
    pd.DataFrame
        ROE signal, same date index as net_income.
    """
    tickers = net_income.columns.intersection(book_equity.columns)
    ni = net_income[tickers]
    be = book_equity[tickers].replace(0, float("nan"))
    roe = ni / be
    return roe
