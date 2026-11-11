"""Rebalance calendar and cost accounting of the long-short builder."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.construction import build_long_short_portfolio, month_end_trading_days  # noqa: E402

IDX = pd.bdate_range("2021-01-01", "2021-12-31")


def test_every_month_has_a_rebalance_date():
    # Jan, Jul and Oct 2021 end on a weekend; they used to be skipped.
    d = month_end_trading_days(IDX)
    assert len(d) == 12
    assert pd.Timestamp("2021-01-29") in d and pd.Timestamp("2021-07-30") in d


def test_cost_charged_once_per_unit_traded():
    rng = np.random.default_rng(0)
    tickers = [f"T{i}" for i in range(100)]
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (len(IDX), 100)), axis=0)), index=IDX, columns=tickers)
    sig = pd.DataFrame(rng.normal(size=prices.shape), index=IDX, columns=tickers)
    out = build_long_short_portfolio({"A": sig}, prices, tc_bps=1e-3)
    first = out.index[0]
    # initial book: 20 longs at +1/20 and 20 shorts at -1/20, turnover = 40 * (1/20) / 2 = 1
    assert out.loc[first, "turnover"] == 1.0
    assert np.isclose(out.loc[first, "gross_ret"] - out.loc[first, "net_ret"], 1e-3)
