"""Shrinkage-coefficient diagnostics for the graphical horseshoe.

The shrinkage coefficient

    kappa_ij = 1 / (1 + lambda_ij^2 * tau^2)

measures how much each off-diagonal is shrunk toward zero.  Values near 0
mean "no shrinkage" (signal, edge present); values near 1 mean "full
shrinkage" (noise, edge absent).

The horseshoe's defining property is that the posterior distribution
of kappa is **bimodal**: most entries cluster near 0 or near 1, with
few in between.  NUTS should preserve this.  Mean-field ADVI is
expected to destroy it by collapsing the joint posterior in the
unconstrained space — this is the paper's central hypothesis.
"""

from typing import Optional

import numpy as np


# ======================================================================
# Kappa computation
# ======================================================================

def compute_kappa_samples(tau_samples, lambda_samples):
    """Compute the posterior samples of the shrinkage coefficient.

    Parameters
    ----------
    tau_samples : np.ndarray, shape (n_samples,)
        Posterior samples of the global shrinkage scalar.
    lambda_samples : np.ndarray, shape (n_samples, n_offdiag)
        Posterior samples of the per-edge local shrinkage parameters.

    Returns
    -------
    kappa_samples : np.ndarray, shape (n_samples, n_offdiag)
    """
    tau = np.asarray(tau_samples).reshape(-1, 1)  # (n_samples, 1)
    lam = np.asarray(lambda_samples)              # (n_samples, n_offdiag)
    if lam.ndim != 2:
        raise ValueError(f"lambda_samples must be 2D, got shape {lam.shape}")
    if tau.shape[0] != lam.shape[0]:
        raise ValueError(
            f"tau/lambda sample count mismatch: {tau.shape[0]} vs {lam.shape[0]}"
        )
    return 1.0 / (1.0 + (lam * tau) ** 2)


def compute_kappa_hat(kappa_samples):
    """Posterior mean of the per-edge shrinkage coefficients."""
    return np.mean(np.asarray(kappa_samples), axis=0)


# ======================================================================
# Bimodality coefficient
# ======================================================================

BIMODALITY_THRESHOLD = 5.0 / 9.0  # ~0.5556 — values above suggest bimodality


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient.

        b = (g^2 + 1) / (k + 3(n-1)^2 / ((n-2)(n-3)))

    where ``g`` is the sample skewness and ``k`` is the sample **excess**
    kurtosis.  The correction term approaches 3 for large n, recovering
    the simpler formula ``b = (g^2 + 1) / k`` used for large samples.

    Values above ``5/9 ≈ 0.556`` suggest the data is bimodal (or more
    generally, not strongly unimodal).  The uniform distribution has
    ``b = 5/9`` exactly.

    Parameters
    ----------
    x : array-like, 1D

    Returns
    -------
    float
        The bimodality coefficient, or NaN if n < 4 or the distribution
        is degenerate (zero variance).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return float("nan")

    mu = np.mean(x)
    diffs = x - mu
    m2 = np.mean(diffs ** 2)
    if m2 == 0:
        return float("nan")
    m3 = np.mean(diffs ** 3)
    m4 = np.mean(diffs ** 4)

    g = m3 / (m2 ** 1.5)           # skewness (biased estimator)
    k = m4 / (m2 ** 2) - 3.0       # excess kurtosis

    numerator = g ** 2 + 1.0
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denominator = k + correction
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


# ======================================================================
# Wasserstein-1 distance between two 1D empirical distributions
# ======================================================================

def shrinkage_wasserstein(kappa_a, kappa_b):
    """Wasserstein-1 distance between two 1D empirical distributions.

    For samples of **equal length**, this reduces to the mean absolute
    difference of sorted samples:

        W_1(a, b) = (1/n) * sum_i |sort(a)_i - sort(b)_i|

    For samples of unequal length, delegates to ``scipy.stats.wasserstein_distance``.
    """
    a = np.asarray(kappa_a, dtype=np.float64).ravel()
    b = np.asarray(kappa_b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")

    if a.size == b.size:
        return float(np.mean(np.abs(np.sort(a) - np.sort(b))))

    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        # Fall back to a simple CDF integral
        return _wasserstein_1d_fallback(a, b)


def _wasserstein_1d_fallback(a, b):
    """Pure-numpy Wasserstein-1 via the integral of |F_a - F_b|."""
    all_values = np.concatenate([a, b])
    grid = np.sort(np.unique(all_values))
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    Fa = np.searchsorted(a_sorted, grid, side="right") / a.size
    Fb = np.searchsorted(b_sorted, grid, side="right") / b.size
    dx = np.diff(grid)
    return float(np.sum(np.abs(Fa[:-1] - Fb[:-1]) * dx))


# ======================================================================
# Shrinkage-profile summary
# ======================================================================

def shrinkage_profile_summary(kappa_hat):
    """Summary statistics of the per-edge shrinkage coefficients.

    Parameters
    ----------
    kappa_hat : array-like
        1D vector of posterior-mean kappa values (one per off-diagonal entry).

    Returns
    -------
    dict
        ``n, mean, median, std, q25, q75, frac_near_0, frac_near_1,
        bimodality_coefficient, is_bimodal``.
    """
    k = np.asarray(kappa_hat, dtype=np.float64).ravel()
    if k.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "frac_near_0": float("nan"),
            "frac_near_1": float("nan"),
            "bimodality_coefficient": float("nan"),
            "is_bimodal": False,
        }
    b = bimodality_coefficient(k)
    return {
        "n": int(k.size),
        "mean": float(np.mean(k)),
        "median": float(np.median(k)),
        "std": float(np.std(k, ddof=0)),
        "q25": float(np.percentile(k, 25)),
        "q75": float(np.percentile(k, 75)),
        "frac_near_0": float(np.mean(k < 0.1)),
        "frac_near_1": float(np.mean(k > 0.9)),
        "bimodality_coefficient": b,
        "is_bimodal": bool(not np.isnan(b) and b > BIMODALITY_THRESHOLD),
    }
