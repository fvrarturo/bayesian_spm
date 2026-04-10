"""Evaluation metrics for precision matrix estimation."""

import numpy as np


def steins_loss(Omega_hat, Omega_true):
    """Stein's loss: tr(Omega_hat^{-1} Omega_true) - log|Omega_hat^{-1} Omega_true| - p.

    Parameters
    ----------
    Omega_hat : np.ndarray, shape (p, p)
    Omega_true : np.ndarray, shape (p, p)

    Returns
    -------
    float
    """
    p = Omega_hat.shape[0]
    M = np.linalg.solve(Omega_hat, Omega_true)
    return np.trace(M) - np.linalg.slogdet(M)[1] - p


def frobenius_loss(Omega_hat, Omega_true):
    """Frobenius norm squared: ||Omega_hat - Omega_true||_F^2.

    Parameters
    ----------
    Omega_hat : np.ndarray, shape (p, p)
    Omega_true : np.ndarray, shape (p, p)

    Returns
    -------
    float
    """
    diff = Omega_hat - Omega_true
    return float(np.sum(diff ** 2))


def spectral_loss(Omega_hat, Omega_true):
    """Spectral (operator) norm: ||Omega_hat - Omega_true||_2.

    Parameters
    ----------
    Omega_hat : np.ndarray, shape (p, p)
    Omega_true : np.ndarray, shape (p, p)

    Returns
    -------
    float
    """
    diff = Omega_hat - Omega_true
    return float(np.linalg.norm(diff, ord=2))


def sparsity_metrics(Omega_hat, Omega_true, threshold=1e-5):
    """Sparsity recovery metrics: TPR, FPR, MCC, F1.

    Evaluates off-diagonal elements only (diagonal is always nonzero).

    Parameters
    ----------
    Omega_hat : np.ndarray, shape (p, p)
        Estimated precision matrix (entries < threshold treated as zero).
    Omega_true : np.ndarray, shape (p, p)
        True precision matrix.
    threshold : float
        Threshold below which entries are considered zero.

    Returns
    -------
    dict
        Keys: "tpr", "fpr", "mcc", "f1", "precision", "recall".
    """
    p = Omega_hat.shape[0]
    idx = np.triu_indices(p, k=1)

    true_nonzero = np.abs(Omega_true[idx]) > threshold
    pred_nonzero = np.abs(Omega_hat[idx]) > threshold

    tp = np.sum(true_nonzero & pred_nonzero)
    fp = np.sum(~true_nonzero & pred_nonzero)
    fn = np.sum(true_nonzero & ~pred_nonzero)
    tn = np.sum(~true_nonzero & ~pred_nonzero)

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
    }
