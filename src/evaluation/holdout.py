"""Out-of-sample evaluation metrics for the WORK4 real-data section.

For real data we don't have a ground-truth precision matrix, so the
synthetic Stein's loss / coverage / bimodality machinery in
``evaluate_single.py`` doesn't apply.  Instead we evaluate Ω̂ against a
held-out window of returns:

- ``oos_nll``: average negative log-likelihood under N(0, Ω̂⁻¹).
- ``gmv_oos_variance``: realised variance of the global-minimum-variance
  portfolio formed from Ω̂, computed on held-out returns.
- ``edge_jaccard``: similarity of credible-interval edge sets across
  windows (Bayesian-only, used to measure structural stability).
- ``condition_number``: numerical conditioning of Ω̂; useful as a
  quality proxy for downstream finance applications.

All functions are pure NumPy.  No JAX, no NumPyro, no scipy.  The intent
is that this module is callable directly from analysis notebooks as well
as from cluster jobs, so it must have a tiny dependency footprint.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


# ======================================================================
# Out-of-sample negative log-likelihood
# ======================================================================

def oos_nll(omega_hat: np.ndarray, Y_test: np.ndarray) -> float:
    """Average negative log-likelihood of held-out data under N(0, Ω̂⁻¹).

    For y ~ N(0, Σ) with Ω = Σ⁻¹, the per-observation log-density is

        log p(y) = -(p/2) log(2π) + (1/2) log|Ω| - (1/2) yᵀ Ω y.

    The OOS NLL averages -log p(y_t) over held-out rows of Y_test.  Lower
    = better fit on unseen data.  Returns ``+inf`` if Ω̂ is not positive
    definite, which lets it stand in as a worst-case score.

    Parameters
    ----------
    omega_hat : (p, p) ndarray
        Estimated precision matrix.
    Y_test : (T, p) ndarray
        Held-out returns (zero-mean assumption — the caller demeans).

    Returns
    -------
    float
        Average negative log-likelihood per row.  ``+inf`` if Ω̂ is not PD.
    """
    omega_hat = np.asarray(omega_hat, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)
    p = omega_hat.shape[0]
    if Y_test.shape[1] != p:
        raise ValueError(
            f"shape mismatch: omega_hat is ({p}, {p}) but Y_test has "
            f"{Y_test.shape[1]} columns"
        )
    # Use Cholesky to (a) reject non-PD matrices unambiguously and (b) get
    # log|Ω| via the diagonal of L cheaply.  ``slogdet`` is unreliable
    # because det(-I_p) = +1 for even p — the sign check would miss
    # negative-definite matrices.
    try:
        L = np.linalg.cholesky(omega_hat)
    except np.linalg.LinAlgError:
        return float("inf")
    logdet = float(2.0 * np.log(np.diag(L)).sum())
    if not np.isfinite(logdet):
        return float("inf")
    # Mean of yᵀ Ω y across rows = mean(Y · Ω · Yᵀ on the diagonal)
    quad_mean = float(np.einsum("ti,ij,tj->", Y_test, omega_hat, Y_test) / Y_test.shape[0])
    return float(0.5 * (p * np.log(2 * np.pi) - logdet + quad_mean))


# ======================================================================
# Global-minimum-variance portfolio
# ======================================================================

def gmv_weights(omega_hat: np.ndarray) -> np.ndarray:
    """Weights of the global minimum variance portfolio under Ω̂.

        w_GMV = (Ω̂ · 1) / (1ᵀ Ω̂ · 1).

    Sums to 1 by construction.  Allows short positions (no non-negativity
    constraint) — the standard textbook GMV.
    """
    omega_hat = np.asarray(omega_hat, dtype=np.float64)
    p = omega_hat.shape[0]
    one = np.ones(p)
    omega_one = omega_hat @ one
    denom = float(one @ omega_one)
    if not np.isfinite(denom) or abs(denom) < 1e-30:
        raise ValueError("GMV denominator (1ᵀ Ω 1) is zero or non-finite")
    return omega_one / denom


def gmv_oos_variance(omega_hat: np.ndarray, Y_test: np.ndarray) -> float:
    """Realised variance of the GMV portfolio under Ω̂, on held-out returns.

        σ²_OOS = w_GMV ᵀ Σ̂_test w_GMV,
        Σ̂_test = (1/T_test) Y_testᵀ Y_test.

    Lower = better.  A precision estimate that gets the conditional
    structure right will pick a portfolio with low realised variance on
    unseen data.

    Returns
    -------
    float
        The realised OOS variance, or ``+inf`` if the portfolio is undefined
        (Ω̂ singular or 1ᵀ Ω̂ 1 = 0).
    """
    omega_hat = np.asarray(omega_hat, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)
    try:
        w = gmv_weights(omega_hat)
    except ValueError:
        return float("inf")
    Sigma_test = (Y_test.T @ Y_test) / Y_test.shape[0]
    return float(w @ Sigma_test @ w)


def gmv_oos_sharpe(
    omega_hat: np.ndarray, Y_test: np.ndarray, annualize: int = 252,
) -> float:
    """Annualised Sharpe ratio of the GMV portfolio on held-out returns.

    Useful as a sanity check (high-quality Ω̂ → tighter portfolio → less
    realised volatility but also typically less variance in returns —
    Sharpe is a single-number summary).
    """
    omega_hat = np.asarray(omega_hat, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)
    try:
        w = gmv_weights(omega_hat)
    except ValueError:
        return float("nan")
    realised = Y_test @ w  # (T_test,)
    mean = realised.mean()
    std = realised.std(ddof=1)
    if std < 1e-30:
        return float("nan")
    return float(np.sqrt(annualize) * mean / std)


# ======================================================================
# Edge-set stability (Bayesian-only)
# ======================================================================

def credible_edge_set(
    omega_samples: np.ndarray, alpha: float = 0.05,
) -> set[tuple[int, int]]:
    """Edge set for which the central (1−α) credible interval excludes 0.

    Parameters
    ----------
    omega_samples : (n_samples, p, p) ndarray
        Posterior samples of the precision matrix.
    alpha : float
        Significance level.  Default 0.05 → 95% credible intervals.

    Returns
    -------
    set of (i, j) tuples with i < j
        Off-diagonal entries whose CI does not contain 0.
    """
    omega_samples = np.asarray(omega_samples, dtype=np.float64)
    n, p, _ = omega_samples.shape
    iu_i, iu_j = np.triu_indices(p, k=1)
    offdiag = omega_samples[:, iu_i, iu_j]  # (n_samples, n_offdiag)
    lower = np.quantile(offdiag, alpha / 2, axis=0)
    upper = np.quantile(offdiag, 1 - alpha / 2, axis=0)
    excludes_zero = (lower > 0) | (upper < 0)
    out: set[tuple[int, int]] = set()
    for k in range(len(iu_i)):
        if excludes_zero[k]:
            out.add((int(iu_i[k]), int(iu_j[k])))
    return out


def edge_jaccard(
    edge_set_a: Iterable[tuple[int, int]],
    edge_set_b: Iterable[tuple[int, int]],
) -> float:
    """Jaccard similarity between two edge sets.

        J(A, B) = |A ∩ B| / |A ∪ B|.

    Returns 1.0 if both sets are empty (vacuously identical).
    """
    a = set(edge_set_a)
    b = set(edge_set_b)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ======================================================================
# Conditioning
# ======================================================================

def condition_number(omega_hat: np.ndarray) -> float:
    """Spectral condition number λ_max / λ_min of Ω̂ (the precision matrix).

    Returns ``+inf`` for non-PD matrices.  A well-conditioned precision
    estimate is a quality proxy for finance applications: badly
    conditioned Ω̂ produces unstable portfolio weights.
    """
    omega_hat = np.asarray(omega_hat, dtype=np.float64)
    eigs = np.linalg.eigvalsh(omega_hat)
    if eigs.min() <= 0:
        return float("inf")
    return float(eigs.max() / eigs.min())


# ======================================================================
# Top-level wrapper
# ======================================================================

def compute_holdout_metrics(
    omega_hat: np.ndarray,
    Y_test: np.ndarray,
    omega_samples: Optional[np.ndarray] = None,
    annualize: int = 252,
) -> dict:
    """Compute all real-data holdout metrics in one call.

    Parameters
    ----------
    omega_hat : (p, p)
        Posterior mean (Bayesian) or point estimate (frequentist) of Ω.
    Y_test : (T_test, p)
        Held-out returns, demeaned.
    omega_samples : (n_samples, p, p), optional
        Posterior samples (Bayesian only).  If provided, the credible
        edge set is computed and stored in the output.
    annualize : int
        Trading days per year, for the Sharpe ratio.

    Returns
    -------
    dict
        Keys: oos_nll, gmv_oos_variance, gmv_oos_sharpe, condition_number,
        and (Bayesian-only) credible_edge_count.
    """
    out = {
        "oos_nll": oos_nll(omega_hat, Y_test),
        "gmv_oos_variance": gmv_oos_variance(omega_hat, Y_test),
        "gmv_oos_sharpe": gmv_oos_sharpe(omega_hat, Y_test, annualize=annualize),
        "condition_number": condition_number(omega_hat),
    }
    if omega_samples is not None:
        edges = credible_edge_set(omega_samples, alpha=0.05)
        out["credible_edge_count"] = len(edges)
        # Store as a sorted list of [i, j] for JSON serialisability.
        out["credible_edges"] = [list(e) for e in sorted(edges)]
    return out
