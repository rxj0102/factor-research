"""Point-in-time S&P 500 membership for the tickers in the price snapshot.

The price snapshot holds today's index members. Using all of them from the first day of the
sample is a look-ahead: a company that joined in 2021 was picked with knowledge of 2021. This
module answers, for each ticker and day, whether the company was in the index on that day.

Sources (Wikipedia, CC BY-SA): the current constituents table, which carries each member's
date added, and the change log. Frozen copies are in data/snapshot.

What this fixes and what it does not: it removes companies before they joined. It cannot bring
back companies that left the index and are not in the price snapshot at all.
"""
from __future__ import annotations

import io
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SNAP = Path(__file__).resolve().parent / "snapshot"
CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
HEADERS = {"User-Agent": "factor-research (github.com/rxj0102/factor-research)"}


def _tables(url: str) -> list[pd.DataFrame]:
    with urlopen(Request(url, headers=HEADERS), timeout=60) as resp:
        return pd.read_html(io.StringIO(resp.read().decode()))


def _norm(t: pd.Series) -> pd.Series:
    return t.astype(str).str.strip().str.replace(".", "-", regex=False)


def refresh() -> None:
    """Download both tables and overwrite the frozen copies."""
    cur = next(t for t in _tables(CURRENT_URL) if "Symbol" in t.columns and "Date added" in t.columns)
    cur = pd.DataFrame({"ticker": _norm(cur["Symbol"]),
                        "date_added": pd.to_datetime(cur["Date added"], errors="coerce")})
    chg = next(t for t in _tables(CHANGES_URL) if isinstance(t.columns, pd.MultiIndex)
               and ("Added", "Ticker") in t.columns)
    chg = pd.DataFrame({"date": pd.to_datetime(chg.iloc[:, 0], errors="coerce"),
                        "added": _norm(chg[("Added", "Ticker")]),
                        "removed": _norm(chg[("Removed", "Ticker")])}).dropna(subset=["date"])
    chg = chg.replace("nan", "").sort_values("date")
    cur.sort_values("ticker").to_csv(SNAP / "sp500_current.csv", index=False, date_format="%Y-%m-%d")
    chg.to_csv(SNAP / "sp500_changes.csv", index=False, date_format="%Y-%m-%d")


def spells(tickers: list[str]) -> pd.DataFrame:
    """First and last day of index membership for each ticker (NaT means open-ended)."""
    cur = pd.read_csv(SNAP / "sp500_current.csv", parse_dates=["date_added"]).set_index("ticker")
    chg = pd.read_csv(SNAP / "sp500_changes.csv", parse_dates=["date"], keep_default_na=False)
    last_added = chg[chg["added"] != ""].groupby("added")["date"].max()
    last_removed = chg[chg["removed"] != ""].groupby("removed")["date"].max()
    rows = []
    for t in tickers:
        if t in cur.index:
            # the constituents table is the better source for current members: it follows renames
            joined, left = cur.loc[t, "date_added"], pd.NaT
        else:
            joined, left = last_added.get(t, pd.NaT), last_removed.get(t, pd.NaT)
        rows.append({"ticker": t, "joined": joined, "left": left, "current": t in cur.index})
    return pd.DataFrame(rows).set_index("ticker")


def membership_mask(dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """True where the ticker was an index member on that day."""
    sp = spells(tickers)
    mask = pd.DataFrame(True, index=dates, columns=tickers)
    for t in tickers:
        joined, left = sp.loc[t, "joined"], sp.loc[t, "left"]
        if pd.notna(joined):
            mask.loc[dates < joined, t] = False
        if pd.notna(left):
            mask.loc[dates >= left, t] = False
    return mask


if __name__ == "__main__":
    refresh()
    print(pd.read_csv(SNAP / "sp500_changes.csv").tail(3))
