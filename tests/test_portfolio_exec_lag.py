"""Execution lag in the long-short portfolio."""
import numpy as np
import pandas as pd

from portfolio.construction import build_long_short_portfolio


def _setup(n=300, k=40, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"T{i}" for i in range(k)]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)), index=idx, columns=cols)
    sig = pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=cols)
    return {"f": sig}, px


def test_lag_one_starts_returns_a_day_later():
    f, px = _setup()
    p0 = build_long_short_portfolio(f, px, exec_lag=0)
    p1 = build_long_short_portfolio(f, px, exec_lag=1)
    d0, d1 = p0.index[0], p1.index[0]
    assert px.index.get_loc(d1) == px.index.get_loc(d0) + 1


def test_new_weights_do_not_earn_the_fill_day():
    """Changing the signal at a rebalance must leave P&L unchanged through the
    fill day (exec_lag=1): until that close the old holdings are still held."""
    f, px = _setup()
    rebal = [d for d in px.resample("ME").last().index if d in px.index]
    d = rebal[4]
    fill = px.index[px.index.get_loc(d) + 1]
    base = build_long_short_portfolio(f, px, exec_lag=1)
    alt_sig = f["f"].copy()
    alt_sig.loc[d:] = -alt_sig.loc[d:]  # flip the signal from d on
    alt = build_long_short_portfolio({"f": alt_sig}, px, exec_lag=1)
    pd.testing.assert_series_equal(base.loc[:fill, "gross_ret"], alt.loc[:fill, "gross_ret"])
    after = base.index[base.index > fill][0]
    assert base.loc[after, "gross_ret"] != alt.loc[after, "gross_ret"]
