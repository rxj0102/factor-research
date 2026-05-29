"""
Main entry point: run the full factor research pipeline.

Usage:
    python run_research.py [--start 2015-01-01] [--end 2024-06-30] [--fast]

--fast: use a smaller ticker set and shorter date range for quick testing.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_research")

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-06-30")
    p.add_argument("--fast", action="store_true", help="Quick run with fewer tickers")
    return p.parse_args()


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return prices.shift(-horizon) / prices - 1


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Last-day-of-month prices; compute 1-month returns."""
    monthly = prices.resample("ME").last()
    return monthly.pct_change()


def main():
    args = parse_args()

    from data import load_prices, load_market_cap, load_fundamentals_simulated, SP500_TICKERS
    from factors import (
        compute_momentum,
        compute_value,
        compute_quality,
        compute_low_volatility,
        compute_reversal,
        compute_size,
    )
    from analytics import (
        ic_summary,
        factor_decay_curve,
        quintile_portfolios,
        fama_macbeth_regression,
        factor_correlation_matrix,
    )
    from analytics.factor_decay import plot_factor_decay
    from analytics.quintile_analysis import plot_quintile_spreads
    from portfolio import build_long_short_portfolio, portfolio_performance

    tickers = SP500_TICKERS[:80] if args.fast else SP500_TICKERS

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    logger.info("Step 1/7 — Loading price data (%s to %s) …", args.start, args.end)
    prices = load_prices(tickers=tickers, start=args.start, end=args.end)
    logger.info("  Prices shape: %s", prices.shape)

    logger.info("Step 1b — Loading market cap …")
    market_cap = load_market_cap(prices=prices, tickers=list(prices.columns),
                                 start=args.start, end=args.end)

    logger.info("Step 1c — Loading fundamentals (simulated) …")
    fundamentals = load_fundamentals_simulated(prices)

    # ------------------------------------------------------------------ #
    # 2. Compute factors
    # ------------------------------------------------------------------ #
    logger.info("Step 2/7 — Computing factors …")
    f_momentum = compute_momentum(prices)
    f_value = compute_value(market_cap, fundamentals["book_value"])
    f_quality = compute_quality(fundamentals["net_income"], fundamentals["book_equity"])
    f_lowvol = compute_low_volatility(prices)
    f_reversal = compute_reversal(prices)
    f_size = compute_size(market_cap)

    factor_dict = {
        "Momentum": f_momentum,
        "Value": f_value,
        "Quality": f_quality,
        "LowVol": f_lowvol,
        "Reversal": f_reversal,
        "Size": f_size,
    }

    # ------------------------------------------------------------------ #
    # 3. IC / ICIR analysis
    # ------------------------------------------------------------------ #
    logger.info("Step 3/7 — IC / ICIR analysis …")
    fwd_rets = {
        "1d": forward_returns(prices, 1),
        "5d": forward_returns(prices, 5),
        "20d": forward_returns(prices, 20),
    }

    ic_rows = []
    for fname, fsig in factor_dict.items():
        summary = ic_summary(fsig, fwd_rets, window=252)
        summary.insert(0, "factor", fname)
        ic_rows.append(summary)

    ic_df = pd.concat(ic_rows, ignore_index=True)
    ic_df.to_csv(OUTPUTS / "ic_icir_summary.csv", index=False)
    logger.info("  Saved ic_icir_summary.csv")

    # ------------------------------------------------------------------ #
    # 4. Factor decay
    # ------------------------------------------------------------------ #
    logger.info("Step 4/7 — Factor decay curves …")
    decay_df = factor_decay_curve(factor_dict, prices)
    fig = plot_factor_decay(decay_df, save_path=str(OUTPUTS / "factor_decay.png"))
    plt.close("all")
    logger.info("  Saved factor_decay.png")

    # ------------------------------------------------------------------ #
    # 5. Quintile analysis
    # ------------------------------------------------------------------ #
    logger.info("Step 5/7 — Quintile portfolio analysis …")
    quintile_dict = {}
    for fname, fsig in factor_dict.items():
        qdf = quintile_portfolios(fsig, prices)
        quintile_dict[fname] = qdf

    fig = plot_quintile_spreads(quintile_dict, save_path=str(OUTPUTS / "quintile_spreads.png"))
    plt.close("all")
    logger.info("  Saved quintile_spreads.png")

    # ------------------------------------------------------------------ #
    # 6. Fama-MacBeth
    # ------------------------------------------------------------------ #
    logger.info("Step 6/7 — Fama-MacBeth regression …")
    monthly_ret = monthly_returns(prices)
    monthly_factors = {
        name: sig.resample("ME").last() for name, sig in factor_dict.items()
    }
    _, fm_summary = fama_macbeth_regression(monthly_factors, monthly_ret)
    fm_summary.reset_index().to_csv(OUTPUTS / "fama_macbeth_results.csv", index=False)
    logger.info("  Saved fama_macbeth_results.csv")

    # Factor correlation
    avg_corr, crowded = factor_correlation_matrix(factor_dict)
    avg_corr.to_csv(OUTPUTS / "factor_correlation.csv")
    if not crowded.empty:
        logger.info("  Crowded pairs:\n%s", crowded.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 7. Portfolio construction & performance
    # ------------------------------------------------------------------ #
    logger.info("Step 7/7 — Building long-short portfolio …")
    port_df = build_long_short_portfolio(factor_dict, prices)
    perf = portfolio_performance(port_df, ret_col="net_ret")

    perf_df = pd.DataFrame([perf])
    perf_df.to_csv(OUTPUTS / "portfolio_performance.csv", index=False)
    logger.info("  Saved portfolio_performance.csv")

    # Print summary
    logger.info("\n" + "=" * 55)
    logger.info("PORTFOLIO PERFORMANCE SUMMARY")
    logger.info("=" * 55)
    for k, v in perf.items():
        if isinstance(v, float):
            logger.info("  %-30s %+.4f", k, v)
        else:
            logger.info("  %-30s %s", k, v)

    logger.info("\nIC/ICIR Summary (20d horizon):")
    ic_20 = ic_df[ic_df["horizon"] == "20d"][["factor", "mean_IC", "ICIR", "IC_positive_pct"]]
    logger.info("\n" + ic_20.to_string(index=False))
    logger.info("=" * 55)
    logger.info("Pipeline complete. Outputs written to ./outputs/")


if __name__ == "__main__":
    main()
