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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def sparsity_metrics(Omega_hat, Omega_true, threshold=1e-5):
    """Sparsity recovery metrics: TPR, FPR, MCC, F1.

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
    raise NotImplementedError
