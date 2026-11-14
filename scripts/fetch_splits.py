"""Stock split history from yfinance for every ticker in the price snapshot.

Yahoo prices are split-adjusted to today's share basis while EDGAR share counts
are as reported, so market cap needs the split history. Writes
data/snapshot/splits.csv (ticker, date, ratio).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

SNAP = Path(__file__).resolve().parents[1] / "data" / "snapshot"


def main() -> None:
    tickers = list(pd.read_csv(SNAP / "prices_2015-01-01_2024-06-30.csv.gz", index_col=0, nrows=1).columns)
    rows, failed = [], []
    for t in tickers:
        for attempt in range(3):
            try:
                s = yf.Ticker(t).splits
                for d, r in s.items():
                    rows.append({"ticker": t, "date": pd.Timestamp(d).tz_localize(None).date(), "ratio": float(r)})
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            failed.append(t)
        time.sleep(0.2)
    pd.DataFrame(rows).to_csv(SNAP / "splits.csv", index=False)
    print(len(rows), "split events;", "failed:", failed)


if __name__ == "__main__":
    main()
