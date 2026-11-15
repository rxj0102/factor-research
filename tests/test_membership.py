"""Point-in-time index membership: a ticker must not count before the company joined."""
import pandas as pd

from data.membership import membership_mask, spells

DATES = pd.bdate_range("2015-01-02", "2024-06-28")


def test_a_later_joiner_is_out_until_its_join_date():
    sp = spells(["AAPL", "TSLA"])
    assert sp.loc["AAPL", "joined"] < pd.Timestamp("2015-01-01")
    assert sp.loc["TSLA", "joined"] == pd.Timestamp("2020-12-21")
    mask = membership_mask(DATES, ["AAPL", "TSLA"])
    assert mask["AAPL"].all()
    assert not mask.loc["2020-12-18", "TSLA"] and mask.loc["2020-12-21", "TSLA"]


def test_a_ticker_that_left_is_out_from_its_removal_date():
    sp = spells(["AAL"])
    assert not sp.loc["AAL", "current"]
    mask = membership_mask(pd.bdate_range("2015-01-02", "2024-12-31"), ["AAL"])
    assert not mask.loc["2015-03-20", "AAL"] and mask.loc["2015-03-23", "AAL"]
    assert mask.loc["2024-09-20", "AAL"] and not mask.loc["2024-09-23", "AAL"]


def test_unknown_ticker_is_treated_as_a_member_throughout():
    """No join or leave record means no evidence to exclude; the README counts these."""
    assert membership_mask(DATES, ["NOT_A_TICKER"])["NOT_A_TICKER"].all()


def test_masking_removes_a_planted_look_ahead():
    """Signals of names that have not joined yet must disappear, and only those."""
    mask = membership_mask(DATES, ["AAPL", "TSLA"])
    signal = pd.DataFrame(1.0, index=DATES, columns=["AAPL", "TSLA"]).where(mask)
    assert signal["TSLA"].loc[:"2020-12-18"].isna().all()
    assert signal["TSLA"].loc["2020-12-21":].notna().all() and signal["AAPL"].notna().all()
