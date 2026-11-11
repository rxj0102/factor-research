"""build_long_short_portfolio must run end to end on a simple panel."""
import numpy as np
import pandas as pd

from portfolio.construction import build_long_short_portfolio


def test_portfolio_builds():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=200)
    cols = [f"T{i}" for i in range(30)]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (200, 30)), axis=0)), index=idx, columns=cols)
    sig = pd.DataFrame(rng.normal(size=(200, 30)), index=idx, columns=cols)
    out = build_long_short_portfolio({"f": sig}, px)
    assert len(out) > 100
    assert out["net_ret"].notna().all()
