# Cross-Sectional Factor Research Framework

[![CI](https://github.com/rxj0102/factor-research/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/factor-research/actions/workflows/ci.yml)

A production-quality quantitative research framework for systematic factor investing on US equities (S&P 500 universe). Implements six canonical factors, rigorous analytics (IC, ICIR, Fama-MacBeth, factor decay, quintile analysis), and a multi-factor long-short portfolio with ERC weighting.

---

## Project Structure

```
factor-research/
├── data/               # Data loading utilities + static S&P 500 universe
│   ├── universe.py     # ~490 S&P 500 tickers
│   └── loader.py       # yfinance price/market-cap downloads; fundamental stubs
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

**Status:** no verified results are published. An earlier version of this section listed approximate IC and Sharpe figures that were not produced by a recorded run, so they have been removed. Run `python run_research.py` (add `--fast` for 80 tickers) and the tables are written to `outputs/` (`ic_icir_summary.csv`, `fama_macbeth_results.csv`, and the portfolio performance summary).

### Caveats that apply to any result

- **Value and Quality use simulated fundamentals.** `run_research.py` calls `load_fundamentals_simulated`, which generates random log-normal book value, net income and equity. The Value (book-to-market) and Quality (ROE) factors are therefore noise and say nothing about real stocks until real fundamentals are supplied. Momentum, Reversal, Low Volatility and Size use real prices and market cap.
- **Survivorship bias.** The universe is a static list of current S&P 500 members.
- **Execution timing.** Portfolio weights set at the rebalance close are now filled at the next close (`exec_lag=1` in `build_long_short_portfolio`), so the first return earned is two trading days after the signal. `exec_lag=0` reproduces the old same-close fill; `run_research.py` writes both to `outputs/exec_lag_comparison.csv`.
- **Multiple testing.** Six factors and several variants were examined; the Deflated Sharpe Ratio is reported for that reason, but it does not remove survivorship bias.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (downloads ~2 GB of data on first run)
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

---

## References

- Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*.
- Fama, E. & French, K. (1992). The Cross-Section of Expected Stock Returns. *Journal of Finance*.
- Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability Premium. *Journal of Financial Economics*.
- Ang, A., et al. (2006). The Cross-Section of Volatility and Expected Returns. *Journal of Finance*.
- Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
- Fama, E. & MacBeth, J. (1973). Risk, Return, and Equilibrium. *Journal of Political Economy*.
