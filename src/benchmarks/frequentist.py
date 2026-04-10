"""Frequentist benchmark estimators for precision/covariance matrices."""

import numpy as np


def run_sample_cov(Y):
    """Compute the sample covariance and its inverse (if invertible).

    Parameters
    ----------
    Y : np.ndarray, shape (T, p)
        Observation matrix.

    Returns
    -------
    Sigma_hat : np.ndarray, shape (p, p)
    Omega_hat : np.ndarray or None
        None if sample covariance is singular.
    """
    raise NotImplementedError


def run_ledoit_wolf(Y):
    """Ledoit-Wolf linear shrinkage estimator.

    Parameters
    ----------
    Y : np.ndarray, shape (T, p)

    Returns
    -------
    Sigma_hat : np.ndarray, shape (p, p)
    Omega_hat : np.ndarray, shape (p, p)
    shrinkage_intensity : float
    """
    raise NotImplementedError


def run_glasso(Y, alphas=None, cv=5):
    """Graphical lasso with cross-validation.

    Parameters
    ----------
    Y : np.ndarray, shape (T, p)
    alphas : array-like or None
        Regularization path. If None, uses GraphicalLassoCV defaults.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    Sigma_hat : np.ndarray, shape (p, p)
    Omega_hat : np.ndarray, shape (p, p)
    alpha_selected : float
    """
    raise NotImplementedError
