"""Point-in-time rules for EDGAR fundamentals."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.fundamentals_pit import as_of_panel, shares_panel  # noqa: E402

DATES = pd.bdate_range("2020-01-01", "2021-12-31")


def _df(rows):
    df = pd.DataFrame(rows, columns=["ticker", "item", "end", "filed", "val"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df.sort_values("filed").drop_duplicates(["ticker", "item", "end"], keep="first")


def test_not_known_before_filing_and_original_beats_restatement():
    df = _df([
        ("X", "equity", "2019-12-31", "2020-02-20", 100.0),
        ("X", "equity", "2019-12-31", "2021-02-20", 999.0),  # restated comparative in a later filing
        ("X", "equity", "2020-12-31", "2021-02-20", 120.0),
    ])
    p = as_of_panel(df, "equity", DATES, ["X"], max_age_days=10_000)["X"]
    assert np.isnan(p.loc["2020-02-20"])          # filed that day: usable only from the next day
    assert p.loc["2020-02-21"] == 100.0
    assert p.loc["2021-02-19"] == 100.0           # the restatement never replaces the original
    assert p.loc["2021-02-22"] == 120.0


def test_comparative_rows_do_not_roll_back_to_older_periods():
    df = _df([
        ("X", "equity", "2020-06-30", "2020-08-01", 110.0),
        ("X", "equity", "2019-06-30", "2020-09-01", 90.0),  # older period first seen in a later filing
    ])
    p = as_of_panel(df, "equity", DATES, ["X"], max_age_days=10_000)["X"]
    assert p.loc["2020-10-01"] == 110.0


def test_stale_values_expire():
    df = _df([("X", "equity", "2019-12-31", "2020-02-20", 100.0)])
    p = as_of_panel(df, "equity", DATES, ["X"], max_age_days=200)["X"]
    assert p.loc["2020-07-01"] == 100.0
    assert np.isnan(p.loc["2020-08-03"])


def test_shares_converted_to_post_split_basis():
    df = _df([("X", "shares", "2020-03-31", "2020-04-20", 1_000.0)])
    splits = pd.DataFrame({"ticker": ["X"], "date": ["2020-08-31"], "ratio": [4.0]})
    p = shares_panel(df, splits, DATES, ["X"], max_age_days=10_000)["X"]
    # split-adjusted prices are on the post-split basis, so reported shares are scaled by 4
    assert p.loc["2020-05-01"] == 4_000.0
