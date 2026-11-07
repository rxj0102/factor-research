"""No-lookahead and sanity tests for the price-based factors and the IC helper."""

import numpy as np
import pandas as pd
import pytest

from analytics.deflated_sharpe import deflated_sharpe_ratio
from analytics.ic_analysis import _cross_sectional_ic
from factors.low_volatility import compute_low_volatility
from factors.momentum import compute_momentum
from factors.reversal import compute_reversal


def _prices(n=400, k=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    rets = rng.normal(0.0003, 0.012, (n, k))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                        columns=[f"T{i}" for i in range(k)])


FACTORS = [compute_momentum, compute_reversal, compute_low_volatility]


@pytest.mark.parametrize("fn", FACTORS)
def test_factor_is_point_in_time(fn):
    """Signal at date t must be identical when computed from data truncated at t."""
    px = _prices()
    full = fn(px)
    for cut in (300, 350):
        trunc = fn(px.iloc[:cut])
        pd.testing.assert_frame_equal(trunc, full.iloc[:cut])


@pytest.mark.parametrize("fn", FACTORS)
def test_factor_shape_matches_prices(fn):
    px = _prices()
    out = fn(px)
    assert out.shape == px.shape
    assert out.index.equals(px.index) and list(out.columns) == list(px.columns)


def test_momentum_skips_most_recent_month_and_needs_history():
    px = _prices()
    m = compute_momentum(px)
    assert m.iloc[:252].isna().all().all()
    t = 300
    expected = px.iloc[t - 21] / px.iloc[t - 252] - 1
    pd.testing.assert_series_equal(m.iloc[t], expected, check_names=False)


def test_reversal_sign_recent_losers_score_higher():
    px = _prices()
    r = compute_reversal(px, window=5)
    t = 100
    five_day = px.iloc[t] / px.iloc[t - 5] - 1
    assert r.iloc[t].idxmax() == five_day.idxmin()


def test_low_vol_prefers_calmer_assets():
    px = _prices()
    px["CALM"] = 100 * (1 + 0.0001 * np.arange(len(px)))
    sig = compute_low_volatility(px, window=20)
    assert sig.iloc[-1].idxmax() == "CALM"


def test_cross_sectional_ic_perfect_and_inverse():
    s = pd.Series(np.arange(20.0), index=list("abcdefghijklmnopqrst"))
    assert _cross_sectional_ic(s, s * 2) == pytest.approx(1.0)
    assert _cross_sectional_ic(s, -s) == pytest.approx(-1.0)


def test_cross_sectional_ic_needs_ten_names():
    s = pd.Series(np.arange(5.0), index=list("abcde"))
    assert np.isnan(_cross_sectional_ic(s, s))


def test_deflated_sharpe_more_trials_never_raises_dsr():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0008, 0.01, 750)
    one = deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    many = deflated_sharpe_ratio(r, n_trials=100)["dsr"]
    assert many <= one


def test_deflated_sharpe_zero_vol_is_handled():
    out = deflated_sharpe_ratio(np.zeros(100), n_trials=5)
    assert out["dsr"] == 0.0 and out["p_value"] == 1.0
