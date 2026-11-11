"""Factor study on real, point-in-time data.

Momentum (12-1), Reversal (5 day) and Low Volatility (20 day) use total-return
prices. Value (book-to-market), Quality (ROE) and Size use SEC EDGAR
fundamentals and share counts, each entering on the day after its filing
(data/fundamentals_pit.py). Market cap is the split-adjusted, dividend-
unadjusted close times reported shares. The composite is the ERC-weighted
long-short portfolio from portfolio/construction.py, net of 5 bps costs.
Universe: current S&P 500 members.

Monthly, non-overlapping evaluation: the signal at each month-end is compared
with the return over the following month.

Writes results/factor_ic.csv, results/quintile_spread.csv,
results/fama_macbeth.csv, results/fama_macbeth_old_alignment.csv (pre-fix
timing, kept as a diagnostic) and results/run_info.json.

Inputs are frozen in data/snapshot/ (prices, split-adjusted closes, EDGAR
facts, splits). Refresh the price snapshot with --refresh; the other inputs
have their own fetch_*.py scripts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics.fama_macbeth import fama_macbeth_regression  # noqa: E402
from data.fundamentals_pit import build as build_fundamentals  # noqa: E402
from data.universe import SP500_TICKERS  # noqa: E402
from factors import (  # noqa: E402
    compute_low_volatility, compute_momentum, compute_quality, compute_reversal, compute_size, compute_value,
)
from portfolio import build_long_short_portfolio, portfolio_performance  # noqa: E402

START, END = "2015-01-01", "2024-06-30"
SNAP = ROOT / "data" / "snapshot"
OUT = ROOT / "results"


def snapshot_prices(refresh: bool) -> pd.DataFrame:
    path = SNAP / f"prices_{START}_{END}.csv.gz"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    frames = []
    for i in range(0, len(SP500_TICKERS), 25):
        batch = SP500_TICKERS[i:i + 25]
        for attempt in range(3):
            raw = yf.download(batch, start=START, end=END, auto_adjust=True, progress=False, threads=False)
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            close = close.dropna(axis=1, how="all")
            if not close.empty:
                frames.append(close)
                break
            time.sleep(10 * (attempt + 1))
        time.sleep(1)
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    SNAP.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, float_format="%.4f", index_label="Date", compression="gzip")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def monthly_ic(signal: pd.DataFrame, me_prices: pd.DataFrame) -> pd.Series:
    """Spearman IC of month-end signal against the next month's return."""
    nxt = me_prices.shift(-1) / me_prices - 1
    sig = signal.reindex(me_prices.index)
    out = {}
    for d in me_prices.index[:-1]:
        x, y = sig.loc[d], nxt.loc[d]
        ok = x.notna() & y.notna()
        if ok.sum() >= 50:
            out[d] = x[ok].rank().corr(y[ok].rank())
    return pd.Series(out)


def quintile_spread(signal: pd.DataFrame, me_prices: pd.DataFrame) -> pd.Series:
    """Equal-weight top-minus-bottom quintile return over the next month."""
    nxt = me_prices.shift(-1) / me_prices - 1
    sig = signal.reindex(me_prices.index)
    out = {}
    for d in me_prices.index[:-1]:
        x, y = sig.loc[d], nxt.loc[d]
        ok = x.notna() & y.notna()
        if ok.sum() < 50:
            continue
        q = pd.qcut(x[ok].rank(method="first"), 5, labels=False)
        out[d] = y[ok][q == 4].mean() - y[ok][q == 0].mean()
    return pd.Series(out)


def tstat(s: pd.Series) -> float:
    return float(s.mean() / s.std(ddof=1) * np.sqrt(len(s)))


def main() -> None:
    refresh = "--refresh" in sys.argv
    prices = snapshot_prices(refresh)
    prices = prices.loc[:, prices.notna().mean() >= 0.5].ffill(limit=5)
    close = pd.read_csv(SNAP / f"close_splitadj_{START}_{END}.csv.gz", index_col=0, parse_dates=True)
    close = close.reindex(index=prices.index, columns=prices.columns).ffill(limit=5)
    fund = build_fundamentals(close)

    factors = {
        "Momentum": compute_momentum(prices),
        "Value": compute_value(fund["market_cap"], fund["book_equity"]),
        "Quality": compute_quality(fund["net_income"], fund["book_equity"]),
        "LowVol": compute_low_volatility(prices),
        "Reversal": compute_reversal(prices),
        "Size": compute_size(fund["market_cap"]),
    }
    me = prices.resample("ME").last()
    me_idx = [prices.index[prices.index <= d][-1] for d in me.index]
    me_prices = prices.loc[me_idx]

    ic_rows, q_rows = [], []
    for name, f in factors.items():
        ic = monthly_ic(f, me_prices)
        ic_rows.append({"factor": name, "months": len(ic), "mean_IC": ic.mean(), "t_stat": tstat(ic),
                        "IC_positive_share": (ic > 0).mean()})
        sp = quintile_spread(f, me_prices)
        q_rows.append({"factor": name, "months": len(sp), "mean_monthly": sp.mean(),
                       "ann_return": sp.mean() * 12, "ann_sharpe": sp.mean() / sp.std(ddof=1) * np.sqrt(12),
                       "t_stat": tstat(sp)})

    monthly_ret = me_prices.pct_change()
    monthly_factors = {n: f.reindex(me_prices.index) for n, f in factors.items()}
    _, fm = fama_macbeth_regression(monthly_factors, monthly_ret)
    # Diagnostic: the pre-fix alignment (signal at t against the month ending at t).
    # The function now shifts returns forward one month, so shift back to emulate it.
    _, fm_old = fama_macbeth_regression(monthly_factors, monthly_ret.shift(1))

    port = build_long_short_portfolio(factors, prices)
    perf = portfolio_performance(port, ret_col="net_ret", n_trials=len(factors))
    perf_gross = portfolio_performance(port, ret_col="gross_ret", n_trials=len(factors))

    OUT.mkdir(exist_ok=True)
    port.to_csv(OUT / "composite_daily_returns.csv", float_format="%.8f", index_label="date")
    (OUT / "composite_performance.json").write_text(json.dumps(
        {"net": {k: float(v) for k, v in perf.items()},
         "gross": {k: float(v) for k, v in perf_gross.items()}}, indent=2) + "\n")
    pd.DataFrame(ic_rows).to_csv(OUT / "factor_ic.csv", index=False, float_format="%.6g")
    pd.DataFrame(q_rows).to_csv(OUT / "quintile_spread.csv", index=False, float_format="%.6g")
    fm.reset_index().to_csv(OUT / "fama_macbeth.csv", index=False, float_format="%.6g")
    fm_old.reset_index().to_csv(OUT / "fama_macbeth_old_alignment.csv", index=False, float_format="%.6g")
    cov = {k: float(v.loc[prices.index >= "2016-01-01"].notna().mean().mean()) for k, v in fund.items()}
    info = {"start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
            "n_tickers": int(prices.shape[1]), "fundamental_coverage": cov,
            "universe": "current S&P 500 members (survivorship biased)"}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print(pd.DataFrame(ic_rows).to_string(index=False))
    print(pd.DataFrame(q_rows).to_string(index=False))
    print(fm.to_string())
    print(json.dumps(perf, indent=2, default=float))


if __name__ == "__main__":
    main()
