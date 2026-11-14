"""Download point-in-time fundamentals from SEC EDGAR XBRL company facts.

For each ticker in the price snapshot, extracts as-reported values with their
filing dates:
  equity  StockholdersEquity (fallback: including noncontrolling interest)
  ni_fy   NetIncomeLoss over annual periods (350 to 380 days)
  shares  dei:EntityCommonStockSharesOutstanding (share classes summed per filing)

Writes data/snapshot/edgar_fundamentals.csv.gz (long format: ticker, item, end,
filed, val). Free, no key; the SEC asks for a User-Agent with contact details
and at most 10 requests per second.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshot"
UA = "Rajvardhan Jadhav rajvardhanjadhav124@gmail.com"


def get(url: str) -> dict:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(url)


def rows(facts: dict, ns: str, tag: str, unit: str, item: str, annual: bool = False) -> list[dict]:
    try:
        recs = facts[ns][tag]["units"][unit]
    except KeyError:
        return []
    out = []
    for r in recs:
        if r.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
            continue
        if annual:
            if "start" not in r:
                continue
            days = (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days
            if not 350 <= days <= 380:
                continue
        out.append({"item": item, "end": r["end"], "filed": r["filed"], "accn": r["accn"], "val": r["val"]})
    return out


def main() -> None:
    prices = pd.read_csv(SNAP / "prices_2015-01-01_2024-06-30.csv.gz", index_col=0, nrows=1)
    tickers = list(prices.columns)
    cmap = {v["ticker"].replace("-", "."): v["cik_str"] for v in get("https://www.sec.gov/files/company_tickers.json").values()}
    cmap.update({k.replace(".", "-"): v for k, v in list(cmap.items())})
    out, missing = [], []
    for i, t in enumerate(tickers):
        cik = cmap.get(t) or cmap.get(t.replace("-", "."))
        if cik is None:
            missing.append(t)
            continue
        try:
            facts = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json")["facts"]
        except RuntimeError:
            missing.append(t)
            continue
        eq = rows(facts, "us-gaap", "StockholdersEquity", "USD", "equity")
        if not eq:
            eq = rows(facts, "us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD", "equity")
        ni = rows(facts, "us-gaap", "NetIncomeLoss", "USD", "ni_fy", annual=True)
        sh = rows(facts, "dei", "EntityCommonStockSharesOutstanding", "shares", "shares")
        if sh:  # several share classes in one filing are summed
            s = pd.DataFrame(sh).groupby(["accn", "end", "filed"], as_index=False)["val"].sum()
            s["item"] = "shares"
            sh = s.to_dict("records")
        for r in eq + ni + sh:
            r["ticker"] = t
            out.append(r)
        time.sleep(0.12)
        if i % 50 == 0:
            print(i, t, len(out), flush=True)
    df = pd.DataFrame(out)[["ticker", "item", "end", "filed", "accn", "val"]]
    df = df.drop_duplicates().sort_values(["ticker", "item", "end", "filed"])
    df.to_csv(SNAP / "edgar_fundamentals.csv.gz", index=False, compression="gzip")
    print("tickers with data:", df.ticker.nunique(), "missing:", missing)


if __name__ == "__main__":
    sys.exit(main())
