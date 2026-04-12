"""Evaluation metrics for precision matrix estimation.

Covers:
- Loss functions (Stein, Frobenius, spectral, trace)
- Sparsity recovery (threshold-based and credible-interval-based)
- Posterior calibration (95% coverage, interval widths)
- Eigenvalue comparison
- GMV portfolio weights

All functions are pure — they take numpy arrays, they return floats or dicts.
No disk I/O, no logging.
"""

from typing import Optional

import numpy as np


# ======================================================================
# Loss functions
# ======================================================================

def steins_loss(Omega_hat, Omega_true):
    """Stein's loss: tr(Omega_hat^{-1} Omega_true) - log|Omega_hat^{-1} Omega_true| - p.

    Equals KL(N(0, Omega_hat^{-1}) || N(0, Omega_true^{-1})) up to a constant.

    Parameters
    ----------
    Omega_hat, Omega_true : np.ndarray, shape (p, p)

    Returns
    -------
    float
        Non-negative.  Equals 0 iff Omega_hat == Omega_true.
    """
    p = Omega_hat.shape[0]
    M = np.linalg.solve(Omega_hat, Omega_true)
    return float(np.trace(M) - np.linalg.slogdet(M)[1] - p)


def frobenius_loss(Omega_hat, Omega_true):
    """Frobenius norm squared: ||Omega_hat - Omega_true||_F^2."""
    diff = Omega_hat - Omega_true
    return float(np.sum(diff ** 2))


def frobenius_loss_relative(Omega_hat, Omega_true):
    """Frobenius norm squared, normalised by ||Omega_true||_F^2."""
    num = frobenius_loss(Omega_hat, Omega_true)
    denom = float(np.sum(Omega_true ** 2))
    if denom == 0:
        return float("nan")
    return num / denom


def spectral_loss(Omega_hat, Omega_true):
    """Spectral (operator) norm: ||Omega_hat - Omega_true||_2.

    Equals the largest singular value of the difference.
    """
    diff = Omega_hat - Omega_true
    return float(np.linalg.norm(diff, ord=2))


def trace_error(Omega_hat, Omega_true):
    """Signed difference in traces: tr(Omega_hat) - tr(Omega_true)."""
    return float(np.trace(Omega_hat) - np.trace(Omega_true))


# ======================================================================
# Sparsity recovery
# ======================================================================

def _binary_metrics(pred_nonzero, true_nonzero):
    """Shared TP/FP/FN/TN → TPR/FPR/MCC/F1 logic."""
    tp = int(np.sum(true_nonzero & pred_nonzero))
    fp = int(np.sum(~true_nonzero & pred_nonzero))
    fn = int(np.sum(true_nonzero & ~pred_nonzero))
    tn = int(np.sum(~true_nonzero & ~pred_nonzero))

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / max(denom, 1e-12)

    return {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "mcc": float(mcc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "n_edges_detected": int(np.sum(pred_nonzero)),
        "n_edges_true": int(np.sum(true_nonzero)),
    }


def sparsity_metrics(Omega_hat, Omega_true, threshold=1e-5):
    """Sparsity recovery metrics via thresholding the estimated precision matrix.

    An edge ``(i, j)`` is declared present iff ``|Omega_hat[i, j]| > threshold``.
    The ground-truth edge set is derived the same way from ``Omega_true``.

    Parameters
    ----------
    Omega_hat, Omega_true : np.ndarray, shape (p, p)
    threshold : float

    Returns
    -------
    dict
        Keys: ``tpr``, ``fpr``, ``mcc``, ``f1``, ``precision``, ``recall``,
        ``n_edges_detected``, ``n_edges_true``.
    """
    p = Omega_hat.shape[0]
    idx = np.triu_indices(p, k=1)
    true_nonzero = np.abs(Omega_true[idx]) > threshold
    pred_nonzero = np.abs(Omega_hat[idx]) > threshold
    return _binary_metrics(pred_nonzero, true_nonzero)


def sparsity_metrics_credible(
    omega_samples,
    Omega_true,
    alpha=0.05,
    true_threshold=1e-5,
):
    """Sparsity recovery metrics via credible-interval edge detection.

    For each off-diagonal (i, j), declare an edge present iff the (1-alpha)
    posterior credible interval for omega_ij excludes zero.  Ground-truth
    edges are still determined by thresholding Omega_true.

    Parameters
    ----------
    omega_samples : np.ndarray, shape (n_samples, p, p)
        Full posterior samples.
    Omega_true : np.ndarray, shape (p, p)
    alpha : float
        Significance level for the credible interval (default 0.05 → 95% CI).
    true_threshold : float
        Threshold for declaring ground-truth entries nonzero.

    Returns
    -------
    dict
        Same keys as ``sparsity_metrics``, plus ``alpha``.
    """
    n_samples, p, _ = omega_samples.shape
    idx = np.triu_indices(p, k=1)

    offdiag_samples = omega_samples[:, idx[0], idx[1]]  # (n_samples, n_offdiag)
    q_lo = np.percentile(offdiag_samples, 100 * alpha / 2, axis=0)
    q_hi = np.percentile(offdiag_samples, 100 * (1 - alpha / 2), axis=0)

    pred_nonzero = (q_lo > 0) | (q_hi < 0)
    true_nonzero = np.abs(Omega_true[idx]) > true_threshold

    out = _binary_metrics(pred_nonzero, true_nonzero)
    out["alpha"] = float(alpha)
    return out


# ======================================================================
# Posterior calibration (Bayesian methods only)
# ======================================================================

def coverage_95(omega_samples, Omega_true):
    """Fraction of off-diagonal entries whose true value falls in the 95% CI.

    Returns a dict with:
    - ``coverage_95``: fraction in [0, 1].  Well-calibrated NUTS → ~0.95.
    - ``mean_interval_width``: average width of the 95% CI across entries.
    - ``mean_posterior_std_offdiag``: average posterior std across entries.
    """
    n_samples, p, _ = omega_samples.shape
    idx = np.triu_indices(p, k=1)
    offdiag_samples = omega_samples[:, idx[0], idx[1]]  # (n_samples, n_offdiag)

    q_lo = np.percentile(offdiag_samples, 2.5, axis=0)
    q_hi = np.percentile(offdiag_samples, 97.5, axis=0)
    true_vals = Omega_true[idx]

    covered = (q_lo <= true_vals) & (true_vals <= q_hi)
    coverage = float(np.mean(covered))
    mean_width = float(np.mean(q_hi - q_lo))
    post_std = np.std(offdiag_samples, axis=0, ddof=0)
    mean_std = float(np.mean(post_std))

    return {
        "coverage_95": coverage,
        "mean_interval_width": mean_width,
        "mean_posterior_std_offdiag": mean_std,
    }


# ======================================================================
# Eigenvalue metrics
# ======================================================================

def eigenvalue_metrics(Omega_hat, Omega_true):
    """Compare sorted eigenvalue spectra.

    Returns ``eigenvalue_mse`` (mean squared error between sorted spectra),
    ``condition_number_hat``, ``condition_number_true``.
    """
    eigs_hat = np.sort(np.linalg.eigvalsh(Omega_hat))[::-1]
    eigs_true = np.sort(np.linalg.eigvalsh(Omega_true))[::-1]
    return {
        "eigenvalue_mse": float(np.mean((eigs_hat - eigs_true) ** 2)),
        "condition_number_hat": float(eigs_hat[0] / eigs_hat[-1]),
        "condition_number_true": float(eigs_true[0] / eigs_true[-1]),
    }


# ======================================================================
# GMV portfolio metrics
# ======================================================================

def gmv_metrics(Omega_hat, Omega_true):
    """Compute oracle-vs-estimate GMV weight comparisons.

    Requires ``src.portfolio.gmv.gmv_weights``.
    """
    # Local import to avoid a module-level dependency.
    from src.portfolio.gmv import gmv_weights

    w_hat = gmv_weights(Omega_hat)
    w_true = gmv_weights(Omega_true)
    return {
        "gmv_weight_norm": float(np.linalg.norm(w_hat)),
        "oracle_gmv_weight_norm": float(np.linalg.norm(w_true)),
        "gmv_weight_l2_diff": float(np.linalg.norm(w_hat - w_true)),
        "gmv_weight_l1_diff": float(np.sum(np.abs(w_hat - w_true))),
    }


# ======================================================================
# Safe wrapper
# ======================================================================

def safe_call(fn, *args, default=None, **kwargs):
    """Call ``fn(*args, **kwargs)``, return ``default`` on exception.

    Used to shield metric computation from matrix-singularity errors
    (e.g. Stein's loss on a rank-deficient sample covariance).
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default
