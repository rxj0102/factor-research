"""
Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014).

DSR corrects the observed Sharpe for:
  - Non-normality (skewness, kurtosis)
  - Multiple testing (number of trials)
  - Length of track record

Reference: Bailey, D. H., & Lopez de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
     Non-Normality, and Backtest Overfitting."
    Journal of Portfolio Management.
"""

import numpy as np
from scipy.stats import norm


def deflated_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
    n_trials: int = 1,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute Deflated Sharpe Ratio.

    Parameters
    ----------
    returns : np.ndarray
        Time series of strategy returns (daily or as-sampled).
    benchmark_sharpe : float
        Expected maximum Sharpe across trials under the null (Sharpe = 0).
        If n_trials > 1, this is estimated from order statistics.
    n_trials : int
        Number of backtested strategies / parameter combinations.
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly).

    Returns
    -------
    dict with keys: observed_sr, annualized_sr, benchmark_sr, dsr, p_value.
    """
    T = len(returns)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    skew = _skewness(returns)
    kurt = _kurtosis(returns)  # excess kurtosis

    if sigma == 0:
        return {"observed_sr": 0.0, "annualized_sr": 0.0, "benchmark_sr": benchmark_sharpe,
                "dsr": 0.0, "p_value": 1.0}

    # Observed Sharpe (per-period)
    sr_hat = mu / sigma

    # Annualized
    sr_annual = sr_hat * np.sqrt(periods_per_year)

    # Expected maximum SR under n_trials trials (Eq. 7 in BLP 2014)
    if n_trials > 1:
        expected_max_z = (
            (1 - np.euler_gamma) * norm.ppf(1 - 1.0 / n_trials)
            + np.euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
        benchmark_sharpe = expected_max_z / np.sqrt(periods_per_year)

    # Variance of estimated Sharpe (Eq. 5)
    var_sr = (1 / T) * (1 + 0.5 * sr_hat**2 - skew * sr_hat + (kurt / 4) * sr_hat**2)
    # DSR = P(SR_true > SR_benchmark)
    numerator = (sr_hat - benchmark_sharpe) * np.sqrt(T - 1)
    denominator = np.sqrt(max(var_sr * (T - 1), 1e-12))
    z = numerator / denominator if denominator > 0 else 0.0
    dsr = norm.cdf(z)

    return {
        "observed_sr": sr_hat,
        "annualized_sr": sr_annual,
        "benchmark_sr": benchmark_sharpe,
        "dsr": dsr,
        "p_value": 1 - dsr,
    }


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mu, sig = np.mean(x), np.std(x, ddof=1)
    if sig == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((x - mu) / sig) ** 3)


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    mu, sig = np.mean(x), np.std(x, ddof=1)
    if sig == 0:
        return 0.0
    raw = np.mean(((x - mu) / sig) ** 4)
    return raw - 3.0
