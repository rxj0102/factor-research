"""
Equal Risk Contribution (ERC) portfolio weighting.

ERC minimises the difference between each asset's marginal risk contribution,
so that every asset contributes equally to total portfolio variance.

Solved via sequential quadratic programming (scipy.optimize.minimize).
"""

import numpy as np
from scipy.optimize import minimize


def equal_risk_contribution_weights(
    cov: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    Compute ERC weights for a given covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        (n × n) covariance matrix.
    tol : float
        Optimisation tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    np.ndarray
        ERC weight vector (sums to 1, all non-negative).
    """
    n = cov.shape[0]
    w0 = np.ones(n) / n  # start from equal weights

    def objective(w):
        """Sum of squared pairwise differences of risk contributions."""
        sigma = np.sqrt(w @ cov @ w)
        if sigma < 1e-12:
            return 0.0
        mrc = cov @ w / sigma  # marginal risk contributions
        rc = w * mrc  # risk contributions
        target = sigma / n
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": max_iter},
    )
    w = result.x
    w = np.clip(w, 0, None)
    w /= w.sum()
    return w
