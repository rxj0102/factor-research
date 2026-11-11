"""
Long-short portfolio construction.

- Combine 6 factors via ERC weights.
- Monthly rebalancing: long top quintile, short bottom quintile.
- Apply 5bps one-way transaction cost on gross turnover.
"""

import numpy as np
import pandas as pd

from .erc import equal_risk_contribution_weights


TC_BPS = 5e-4  # 5 basis points one-way


def _composite_signal(
    factor_dict: dict,
    weights: np.ndarray,
    date: pd.Timestamp,
) -> pd.Series:
    """Weighted combination of factor z-scores at a given date."""
    factor_names = list(factor_dict.keys())
    signals = []
    for name in factor_names:
        fdf = factor_dict[name]
        if date not in fdf.index:
            signals.append(pd.Series(dtype=float))
            continue
        s = fdf.loc[date].dropna()
        sd = s.std()  # a scalar; the old .replace(0, nan) call on it raised, so this function never ran
        z = (s - s.mean()) / sd if sd > 0 else s * np.nan
        signals.append(z)

    combined = pd.concat(
        [w * s for w, s in zip(weights, signals) if len(s) > 0], axis=1
    ).sum(axis=1)
    return combined


def _factor_spread_returns(factor_dict: dict, prices: pd.DataFrame, dates: list) -> pd.DataFrame:
    """Top-minus-bottom quintile return of each factor from dates[k] to dates[k+1].

    Each return is realised by dates[k+1], so using these up to the current
    rebalance date is causal.
    """
    out = {}
    for name, fdf in factor_dict.items():
        vals = {}
        for a, b in zip(dates[:-1], dates[1:]):
            if a not in fdf.index:
                continue
            sig = fdf.loc[a].dropna()
            r = (prices.loc[b] / prices.loc[a] - 1).reindex(sig.index).dropna()
            sig = sig.reindex(r.index)
            if len(sig) < 50:
                continue
            k = max(1, int(len(sig) * 0.2))
            vals[b] = r[sig.nlargest(k).index].mean() - r[sig.nsmallest(k).index].mean()
        out[name] = pd.Series(vals)
    return pd.DataFrame(out)


def _erc_factor_weights(factor_dict: dict, lookback_dates: list, prices: pd.DataFrame) -> np.ndarray:
    """ERC weights from the covariance of trailing factor long-short returns.

    An earlier version used the covariance of each factor's cross-sectional
    mean signal level, which is not a return and says nothing about risk.
    """
    n = len(factor_dict)
    if len(lookback_dates) < 7:
        return np.ones(n) / n
    rets = _factor_spread_returns(factor_dict, prices, list(lookback_dates)).dropna()
    if len(rets) < 6:
        return np.ones(n) / n
    rets = rets.reindex(columns=list(factor_dict.keys()))
    cov = rets.cov().values + np.eye(n) * 1e-8  # regularise
    try:
        return equal_risk_contribution_weights(cov)
    except Exception:
        return np.ones(n) / n


def month_end_trading_days(index: pd.DatetimeIndex) -> list:
    """Last trading day of each month. Calendar month-ends that fall on a
    weekend or holiday are not in the price index, and filtering on them
    silently skipped those months."""
    s = index.to_series()
    return list(s.groupby(index.to_period("M")).max())


def build_long_short_portfolio(
    factor_dict: dict,
    prices: pd.DataFrame,
    quintile_cutoff: float = 0.2,
    rebal_freq: str = "ME",
    tc_bps: float = TC_BPS,
) -> pd.DataFrame:
    """
    Build ERC-combined long-short portfolio.

    Parameters
    ----------
    factor_dict : dict
        Factor signal DataFrames (dates × tickers).
    prices : pd.DataFrame
        Adjusted close prices (dates × tickers).
    quintile_cutoff : float
        Fraction of universe in long/short leg (0.2 = top/bottom 20%).
    rebal_freq : str
        Pandas resample offset alias.
    tc_bps : float
        One-way transaction cost per unit traded (5bps = 5e-4).

    Returns
    -------
    pd.DataFrame
        Columns: date, gross_ret, net_ret, long_weight, short_weight, turnover.
    """
    daily_ret = prices.pct_change()
    if rebal_freq != "ME":
        raise ValueError("only month-end rebalancing is supported")
    rebal_dates = month_end_trading_days(prices.index)

    prev_long_w = pd.Series(dtype=float)
    prev_short_w = pd.Series(dtype=float)
    records = []

    for i, d in enumerate(rebal_dates):
        # ERC weights: use previous 12 rebal dates as lookback
        lookback = rebal_dates[max(0, i - 12) : i + 1]  # returns realised up to d
        erc_w = _erc_factor_weights(factor_dict, list(lookback), prices)

        composite = _composite_signal(factor_dict, erc_w, d)
        if composite.empty or composite.dropna().shape[0] < 20:
            continue

        composite = composite.dropna()
        n = len(composite)
        top_n = max(1, int(n * quintile_cutoff))

        long_tickers = composite.nlargest(top_n).index
        short_tickers = composite.nsmallest(top_n).index

        long_w = pd.Series(1.0 / top_n, index=long_tickers)
        short_w = pd.Series(-1.0 / top_n, index=short_tickers)

        # Compute turnover vs previous holdings
        all_tickers = long_w.index.union(short_w.index).union(
            prev_long_w.index).union(prev_short_w.index)
        cur = long_w.reindex(all_tickers, fill_value=0) + short_w.reindex(all_tickers, fill_value=0)
        prev = prev_long_w.reindex(all_tickers, fill_value=0) + prev_short_w.reindex(all_tickers, fill_value=0)
        turnover = (cur - prev).abs().sum() / 2

        # Compute return for the period until next rebal
        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else prices.index[-1]
        period_dates = prices.index[(prices.index > d) & (prices.index <= next_d)]

        if len(period_dates) == 0:
            continue

        period_ret = daily_ret.loc[period_dates]
        long_ret = period_ret[long_tickers].mean(axis=1)
        short_ret = period_ret[short_tickers].mean(axis=1)
        gross_daily = (long_ret - short_ret) / 2  # dollar-neutral scaling

        net_daily = gross_daily.copy()
        # Book is scaled to 0.5 long and 0.5 short, so traded notional per unit
        # capital is sum|dw| / 2 = turnover; both buys and sells are already in
        # it. The earlier version multiplied by 2 again.
        net_daily.iloc[0] -= tc_bps * turnover

        for pd_date in period_dates:
            records.append(
                {
                    "date": pd_date,
                    "gross_ret": gross_daily.get(pd_date, np.nan),
                    "net_ret": net_daily.get(pd_date, np.nan),
                    "turnover": turnover if pd_date == period_dates[0] else 0.0,
                }
            )

        prev_long_w = long_w
        prev_short_w = short_w

    if not records:
        return pd.DataFrame(columns=["date", "gross_ret", "net_ret", "turnover"])

    return pd.DataFrame(records).set_index("date")
