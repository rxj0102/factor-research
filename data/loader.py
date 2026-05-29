"""
Data loading utilities.

Downloads price and market-cap data from yfinance.
Provides stub loaders for fundamental data (B/M, ROE) with simulated
values when real API is unavailable — swap in Simfin/EDGAR calls here.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import SP500_TICKERS

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "processed"
CACHE_DIR.mkdir(exist_ok=True)


def load_prices(
    tickers: list = None,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance.

    Returns DataFrame (dates × tickers) with trading-day index.
    """
    if tickers is None:
        tickers = SP500_TICKERS

    cache_path = CACHE_DIR / f"prices_{start}_{end}.parquet"
    if use_cache and cache_path.exists():
        logger.info("Loading prices from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Downloading prices for %d tickers from yfinance …", len(tickers))
    # Download in batches to avoid rate limiting
    batch_size = 50
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]] if "Close" in raw.columns else raw
            frames.append(close)
        except Exception as exc:
            logger.warning("Failed batch %d: %s", i, exc)
        time.sleep(0.5)

    if not frames:
        raise RuntimeError("No price data downloaded.")

    prices = pd.concat(frames, axis=1)
    prices = prices.sort_index()
    # Drop tickers with >50% missing
    prices = prices.loc[:, prices.isna().mean() < 0.5]
    prices = prices.ffill().dropna(how="all")

    if use_cache:
        prices.to_parquet(cache_path)
    return prices


def load_market_cap(
    prices: pd.DataFrame,
    tickers: list = None,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download shares outstanding via yfinance and multiply by price.

    Falls back to simulated market-cap when shares data is unavailable.
    """
    if tickers is None:
        tickers = list(prices.columns)

    cache_path = CACHE_DIR / f"market_cap_{start}_{end}.parquet"
    if use_cache and cache_path.exists():
        logger.info("Loading market cap from cache.")
        mc = pd.read_parquet(cache_path)
        return mc.reindex(columns=tickers)

    logger.info("Estimating market cap …")
    shares_dict = {}
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).fast_info
            shares = getattr(info, "shares", None) or getattr(info, "sharesOutstanding", None)
            if shares and shares > 0:
                shares_dict[tkr] = shares
        except Exception:
            pass
        time.sleep(0.02)

    if shares_dict:
        shares_series = pd.Series(shares_dict)
        mc = prices[shares_series.index] * shares_series
    else:
        # Fallback: proxy using price * 1e8 (uniform large-cap)
        logger.warning("Shares data unavailable; using price proxy for market cap.")
        mc = prices * 1e8

    mc = mc.sort_index()
    if use_cache:
        mc.to_parquet(cache_path)
    return mc


def load_fundamentals_simulated(
    prices: pd.DataFrame,
    seed: int = 42,
) -> dict:
    """
    Generate realistic but synthetic fundamental data.

    Used when Simfin/EDGAR data is unavailable (free-tier limits).
    Produces quarterly snapshots forward-filled to daily frequency.

    Returns
    -------
    dict with keys:
        'book_value'  : pd.DataFrame (dates × tickers)
        'net_income'  : pd.DataFrame (dates × tickers)
        'book_equity' : pd.DataFrame (dates × tickers)
    """
    rng = np.random.default_rng(seed)
    tickers = list(prices.columns)
    dates = prices.index

    # Quarterly dates
    q_dates = pd.date_range(dates[0], dates[-1], freq="QS")

    def make_panel(base_mean, base_std, trend=0.01):
        # Each ticker gets its own level + random walk
        base = rng.lognormal(mean=np.log(base_mean), sigma=base_std, size=len(tickers))
        shock = rng.normal(trend, 0.05, size=(len(q_dates), len(tickers)))
        panel = base[None, :] * np.exp(shock.cumsum(axis=0))
        df = pd.DataFrame(panel, index=q_dates, columns=tickers)
        # Forward fill to daily
        return df.reindex(dates, method="ffill")

    book_value = make_panel(base_mean=5e9, base_std=1.5)
    net_income = make_panel(base_mean=5e8, base_std=1.2, trend=0.008)
    book_equity = book_value * rng.uniform(0.4, 0.9, size=len(tickers))
    book_equity = pd.DataFrame(
        book_equity.values if isinstance(book_equity, pd.DataFrame) else book_equity,
        index=dates,
        columns=tickers,
    )

    return {
        "book_value": book_value,
        "net_income": net_income,
        "book_equity": book_equity,
    }
