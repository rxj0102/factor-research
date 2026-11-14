# Cross-Sectional Factor Research Framework

[![CI](https://github.com/rxj0102/factor-research/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/factor-research/actions/workflows/ci.yml)

A production-quality quantitative research framework for systematic factor investing on US equities (S&P 500 universe). Implements six canonical factors, rigorous analytics (IC, ICIR, Fama-MacBeth, factor decay, quintile analysis), and a multi-factor long-short portfolio with ERC weighting.

---

## Project Structure

```
factor-research/
├── data/               # Data loading utilities + static S&P 500 universe
│   ├── universe.py     # ~490 S&P 500 tickers
│   ├── loader.py       # yfinance price/market-cap downloads; fundamental stubs
│   └── fundamentals_pit.py  # point-in-time EDGAR panels (book equity, net income, shares)
├── factors/            # One file per factor signal
│   ├── momentum.py     # 12-1 month return (Jegadeesh-Titman)
│   ├── value.py        # Book-to-Market ratio
│   ├── quality.py      # Return on Equity (ROE)
│   ├── low_volatility.py  # Negative 20-day realized vol
│   ├── reversal.py     # Negative 5-day return
│   └── size.py         # Negative log market cap
├── analytics/          # Statistical analysis tools
│   ├── ic_analysis.py      # Rolling IC, ICIR (Spearman)
│   ├── factor_decay.py     # IC decay at lags 1/2/3/5/10/20d
│   ├── quintile_analysis.py # Quintile portfolio returns
│   ├── fama_macbeth.py     # Two-pass FM regression + Newey-West
│   ├── factor_correlation.py # 6×6 monthly corr matrix; crowding flags
│   └── deflated_sharpe.py  # Bailey-Lopez de Prado (2014) DSR
├── portfolio/          # Portfolio construction
│   ├── erc.py          # Equal Risk Contribution optimizer (SQP)
│   ├── construction.py # Long-short with ERC weights + TC
│   └── performance.py  # Annualized return, Sharpe, MDD, DSR
├── outputs/            # Generated charts and CSVs
├── notebooks/
│   └── factor_research_summary.ipynb  # End-to-end walkthrough
├── scripts/
│   ├── real_factor_study.py  # verified study on real point-in-time data
│   ├── fetch_edgar_fundamentals.py, fetch_splits.py, fetch_close_unadjusted.py
│   └── render_readme.py      # renders the Results block, checked in CI
├── results/            # outputs of real_factor_study.py
├── data/snapshot/      # frozen prices and share counts
├── run_research.py     # Main pipeline script
└── requirements.txt
```

---

## Methodology

### Universe
Static list of ~490 S&P 500 constituents. Prices and market cap downloaded via `yfinance`. Fundamentals (B/M, ROE) loaded from Simfin free tier or simulated via realistic log-normal process when API unavailable.

### Factors

| Factor | Signal | Direction | Literature |
|---|---|---|---|
| **Momentum** | Cumulative return months 2–12 | Long winners | Jegadeesh & Titman (1993) |
| **Value** | Book-to-Market ratio | Long cheap | Fama & French (1992) |
| **Quality** | Return on Equity (ROE) | Long profitable | Novy-Marx (2013) |
| **Low Volatility** | −(20d realized vol) | Long low-vol | Ang et al. (2006) |
| **Reversal** | −(5d return) | Long recent losers | Jegadeesh (1990) |
| **Size** | −log(market cap) | Long small | Banz (1981) |

All signals are cross-sectionally z-scored before combination.

### Analytics

- **IC / ICIR**: Spearman rank correlation between factor signal and forward returns (1d, 5d, 20d). Rolling 252-day window.
- **Factor Decay**: Mean IC at forward lags {1, 2, 3, 5, 10, 20} days — measures signal persistence.
- **Quintile Analysis**: Monthly ranking into 5 equal buckets; equal-weighted 20d forward returns; Q5-Q1 spread.
- **Fama-MacBeth**: Monthly cross-sectional OLS of returns on standardized factor exposures; Newey-West t-statistics with 4 lags.
- **Factor Correlation**: Average monthly Spearman correlation matrix; |corr| > 0.6 flagged as crowded.
- **Deflated Sharpe Ratio**: Corrects observed SR for non-normality (skewness, excess kurtosis) and multiple testing (Bailey & Lopez de Prado 2014).

### Portfolio Construction

- **Combination**: Equal Risk Contribution (ERC) weights on all 6 factors; solved via Sequential Quadratic Programming.
- **Long-Short**: Long top quintile (20%), short bottom quintile (20%), dollar-neutral.
- **Rebalancing**: Monthly (end-of-month).
- **Transaction Costs**: 5 bps one-way on gross turnover.

---

## Results

All six factors now run on real data. Momentum, Reversal and Low Volatility use total-return prices. Value (book-to-market), Quality (ROE) and Size come from SEC EDGAR XBRL filings: book equity, fiscal-year net income and shares outstanding, each as first reported and usable from the day after its filing date (`data/fundamentals_pit.py`). Firms with non-positive book equity are excluded from Value and Quality. Market cap is the split-adjusted, dividend-unadjusted close times reported shares, put on a common split basis. `scripts/real_factor_study.py` produces everything below from frozen inputs in `data/snapshot/` and writes `results/`; CI checks that this block matches.

<!-- results:start (generated by scripts/render_readme.py, do not edit by hand) -->
466 current S&P 500 members, 2015-01-02 to 2024-06-28. Each month-end signal is evaluated against the following month's return (non-overlapping). IC t-stats use the monthly IC series; Fama-MacBeth t-stats use Newey-West with 4 lags; the quintile spread is equal-weighted top minus bottom quintile, before costs.

| Factor | Months | Mean IC | IC t-stat | Q5 minus Q1 (ann.) | Spread t-stat | Fama-MacBeth coef | FM t-stat |
|---|---|---|---|---|---|---|---|
| Momentum | 101 | -0.0112 | -0.54 | -1.53% | -0.28 | +0.0025 | +2.07 |
| Value | 113 | -0.0190 | -1.11 | -3.58% | -0.79 | -0.0000 | -0.02 |
| Quality | 113 | +0.0076 | +0.72 | -1.98% | -0.71 | +0.0002 | +0.26 |
| LowVol | 112 | -0.0123 | -0.62 | -9.47% | -1.69 | -0.0039 | -3.17 |
| Reversal | 113 | +0.0144 | +0.98 | +3.22% | +0.73 | +0.0006 | +0.53 |
| Size | 113 | +0.0394 | +3.66 | +12.49% | +3.95 | +0.0022 | +3.13 |

Composite long-short portfolio (ERC-weighted across all six factors, top and bottom quintile, monthly):

| | Gross | Net of 5 bps costs |
|---|---|---|
| Annual return | +2.27% | +1.71% |
| Annual volatility | 5.44% | 5.44% |
| Sharpe ratio | 0.42 | 0.32 |
| Max drawdown | -14.91% | -15.16% |
| Deflated Sharpe (probability SR exceeds the 6-trial benchmark) | 0.004 | 0.002 |

Annual turnover 10.9x. Point-in-time fundamental coverage of the stock-day panel from 2016: book equity 87%, net income 86%, market cap 90%.

Pre-fix Fama-MacBeth timing on the same data, for comparison: Reversal t-stat -27.4 (the signal was regressed on the month whose prices it was built from).
<!-- results:end -->

How to read this:

- **Size is not evidence of a small-cap premium.** The universe is today's S&P 500, so stocks that were small in 2015 and are members now are the ones that grew into the index. Market cap now uses point-in-time share counts, but the survivorship bias remains. Treat this row as a demonstration of that bias.
- **No other factor shows a reliable single-factor premium here.** Momentum, Value, Quality, Low Volatility and Reversal all have IC t-stats inside plus or minus 2, and the Low Volatility quintile spread is negative.
- **Fama-MacBeth tells a different story from the univariate tests.** Controlling for the other factors, Momentum and Low Volatility are significant, with Low Volatility of the opposite sign to the usual anomaly. Six factors on one survivorship-biased sample is a multiple-testing problem, so treat these as exploratory.
- **The composite portfolio does not clear the deflated Sharpe hurdle,** and it includes the survivorship-biased Size factor.

**Fixes made while producing these results.**

- Fama-MacBeth timing: the regression at month-end t used the return over the month ending at t, so a signal built from that month's prices could explain its own inputs. The comparison line under the table shows the effect on Reversal. The signal at t is now regressed on the return from t to t+1 (`analytics/fama_macbeth.py`, with tests that fail on the old alignment). This also affects `run_research.py` and the notebook.
- The composite portfolio could never run: `s.std().replace(0, nan)` called `replace` on a scalar and raised.
- Rebalance dates were calendar month-ends kept only if they were trading days, so months ending on a weekend or holiday were skipped and the prior book was held for two months. Both the portfolio builder and the quintile analysis now use the last trading day of every month.
- The ERC factor weights came from the covariance of each factor's average signal level, which is not a return. They now use trailing 12-month factor long-short returns.
- Costs were charged twice: turnover already counts buys and sells, and was then multiplied by 2 again.

Each of these has a test in `tests/`.

### Caveats that apply to any result

- **`run_research.py` still uses simulated fundamentals.** It calls `load_fundamentals_simulated`, which generates random book value, net income and equity, so its Value and Quality factors are noise. The verified results above come from `scripts/real_factor_study.py`, which uses EDGAR data instead.
- **Survivorship bias.** The universe is a static list of current S&P 500 members.
- **Execution timing.** Signals use the close on the rebalance date and returns start the next day, i.e. a same-close fill. This is optimistic.
- **Multiple testing.** Six factors and several variants were examined; the Deflated Sharpe Ratio is reported for that reason, but it does not remove survivorship bias.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verified study on real point-in-time data (reads data/snapshot/, about 30 seconds)
make reproduce

# Full pipeline including the simulated Value and Quality factors
python run_research.py

# Fast mode (80 tickers, faster)
python run_research.py --fast

# Launch notebook
jupyter notebook notebooks/factor_research_summary.ipynb
```

---

## Key Outputs

| File | Description |
|---|---|
| `outputs/ic_icir_summary.csv` | IC & ICIR per factor per horizon |
| `outputs/factor_decay.png` | 6-panel IC decay curves |
| `outputs/quintile_spreads.png` | 6-panel quintile return bars |
| `outputs/fama_macbeth_results.csv` | FM coefficients + Newey-West t-stats |
| `outputs/factor_correlation.csv` | 6×6 factor correlation matrix |
| `outputs/portfolio_performance.csv` | Full performance metrics |
| `notebooks/factor_research_summary.ipynb` | Interactive end-to-end walkthrough |
| `results/factor_ic.csv`, `results/quintile_spread.csv`, `results/fama_macbeth.csv` | Verified real-price factor study behind the Results table |

---

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)
- SEC EDGAR XBRL company facts (free, requires a User-Agent header)
- `data/snapshot/`, downloaded on 2026-11-10: total-return prices and split-adjusted closes (475 of 504 listed tickers returned data; 466 have at least half the history), EDGAR facts for all 475 (`scripts/fetch_edgar_fundamentals.py`, about 5 minutes), and split history (`scripts/fetch_splits.py`)

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (21 tests) finishes in a few seconds on the machine above. `make reproduce` runs the factor study from the snapshot in about 30 seconds; refreshing the snapshot (`uv run python scripts/real_factor_study.py --refresh`) takes about 7 minutes.

```bash
make reproduce   # factor study from data/snapshot/, write results/, refresh the README block
make check       # confirm the README matches results/ (also run in CI)
```

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## References

- Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*.
- Fama, E. & French, K. (1992). The Cross-Section of Expected Stock Returns. *Journal of Finance*.
- Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability Premium. *Journal of Financial Economics*.
- Ang, A., et al. (2006). The Cross-Section of Volatility and Expected Returns. *Journal of Finance*.
- Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
- Fama, E. & MacBeth, J. (1973). Risk, Return, and Equilibrium. *Journal of Political Economy*.
