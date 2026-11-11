"""Split-adjusted, dividend-unadjusted closes (yfinance auto_adjust=False 'Close').

Total-return prices (auto_adjust=True) shrink past prices by cumulative
dividends, which would understate historical market cap for high-yield stocks.
Market cap therefore uses this series. Writes
data/snapshot/close_splitadj_2015-01-01_2024-06-30.csv.gz.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

SNAP = Path(__file__).resolve().parents[1] / "data" / "snapshot"
START, END = "2015-01-01", "2024-06-30"


def main() -> None:
    tickers = list(pd.read_csv(SNAP / f"prices_{START}_{END}.csv.gz", index_col=0, nrows=1).columns)
    frames = []
    for i in range(0, len(tickers), 25):
        batch = tickers[i:i + 25]
        for attempt in range(3):
            raw = yf.download(batch, start=START, end=END, auto_adjust=False, progress=False, threads=False)
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            close = close.dropna(axis=1, how="all")
            if not close.empty:
                frames.append(close)
                break
            time.sleep(10 * (attempt + 1))
        time.sleep(1)
    df = pd.concat(frames, axis=1).sort_index()
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(SNAP / f"close_splitadj_{START}_{END}.csv.gz", float_format="%.4f", index_label="Date", compression="gzip")
    print(df.shape)


if __name__ == "__main__":
    main()
