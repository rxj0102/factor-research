"""Fama-MacBeth must regress month-t signals on the month t+1 return."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics.fama_macbeth import fama_macbeth_regression  # noqa: E402


def test_contemporaneous_signal_has_no_predictive_power():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    tickers = [f"T{i}" for i in range(80)]
    ret = pd.DataFrame(rng.normal(0, 0.05, (60, 80)), index=dates, columns=tickers)
    # The signal is the same month's return: known only after the fact.
    _, summary = fama_macbeth_regression({"Same": ret.copy()}, ret)
    t = float(summary.loc["Same", "t_stat"])
    # Old alignment gave t-stats in the hundreds; i.i.d. returns give |t| small.
    assert abs(t) < 4


def test_predictive_signal_is_detected():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    tickers = [f"T{i}" for i in range(80)]
    sig = pd.DataFrame(rng.normal(size=(60, 80)), index=dates, columns=tickers)
    ret = 0.02 * sig.shift(1) + rng.normal(0, 0.05, (60, 80))
    _, summary = fama_macbeth_regression({"Pred": sig}, ret)
    assert summary.loc["Pred", "t_stat"] > 5
