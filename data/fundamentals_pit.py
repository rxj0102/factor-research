"""Point-in-time fundamentals from the EDGAR snapshot.

Every value enters the panel on the trading day after its filing date, using
the figure as first reported (later restatements are ignored), so no signal
uses information that was not public at the time.

  book_equity  latest reported StockholdersEquity; NaN when <= 0 (Fama-French)
  net_income   latest reported fiscal-year NetIncomeLoss
  shares       latest reported common shares outstanding, converted to the
               share basis of the split-adjusted price series
  market_cap   split-adjusted (not dividend-adjusted) close times shares

Values older than max_age_days (by period end) are dropped rather than carried
forward indefinitely.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SNAP = Path(__file__).resolve().parent / "snapshot"


def load_edgar(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or SNAP / "edgar_fundamentals.csv.gz", parse_dates=["end", "filed"])
    # as first reported: for each (ticker, item, period end) keep the earliest filing
    return df.sort_values("filed").drop_duplicates(["ticker", "item", "end"], keep="first")


def as_of_panel(df: pd.DataFrame, item: str, dates: pd.DatetimeIndex, tickers: list[str],
                max_age_days: int) -> pd.DataFrame:
    """Value known on each date: the most recent period end among filings made before that date."""
    out = {}
    sub = df[df["item"] == item]
    for t, g in sub.groupby("ticker"):
        if t not in tickers:
            continue
        g = g.sort_values(["filed", "end"])
        # running "latest period end seen so far" in filing order
        best_end = g["end"].cummax()
        g = g[g["end"] == best_end]
        g = g.drop_duplicates("filed", keep="last")
        known_from = g["filed"] + pd.Timedelta(days=1)  # usable the day after filing
        val = pd.Series(g["val"].values, index=known_from)
        end = pd.Series(g["end"].values, index=known_from)
        val = val[~val.index.duplicated(keep="last")]
        end = end[~end.index.duplicated(keep="last")]
        v = val.reindex(dates, method="ffill")
        e = end.reindex(dates, method="ffill")
        stale = (dates.to_series() - e).dt.days > max_age_days
        out[t] = v.where(~stale.values)
    return pd.DataFrame(out, index=dates).reindex(columns=tickers)


def split_factor_after(splits: pd.DataFrame, ticker: str, when: pd.Series) -> np.ndarray:
    """Product of split ratios strictly after each date in `when` (to today's share basis)."""
    s = splits[splits["ticker"] == ticker]
    if s.empty:
        return np.ones(len(when))
    d = pd.to_datetime(s["date"]).values
    r = s["ratio"].values
    w = pd.to_datetime(when).values
    return np.array([r[d > x].prod() if not pd.isna(x) else np.nan for x in w])


def shares_panel(df: pd.DataFrame, splits: pd.DataFrame, dates: pd.DatetimeIndex,
                 tickers: list[str], max_age_days: int = 200) -> pd.DataFrame:
    """Reported shares converted to today's split basis, as known on each date."""
    sub = df[df["item"] == "shares"].copy()
    adj = []
    for t, g in sub.groupby("ticker"):
        g = g.copy()
        g["val"] = g["val"].values * split_factor_after(splits, t, g["end"])
        adj.append(g)
    sub = pd.concat(adj) if adj else sub
    return as_of_panel(sub, "shares", dates, tickers, max_age_days)


def build(prices_close_splitadj: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates, tickers = prices_close_splitadj.index, list(prices_close_splitadj.columns)
    df = load_edgar()
    splits = pd.read_csv(SNAP / "splits.csv")
    be = as_of_panel(df, "equity", dates, tickers, max_age_days=200)
    be = be.where(be > 0)
    ni = as_of_panel(df, "ni_fy", dates, tickers, max_age_days=450)
    sh = shares_panel(df, splits, dates, tickers)
    mcap = prices_close_splitadj * sh
    return {"book_equity": be, "net_income": ni, "shares": sh, "market_cap": mcap}
