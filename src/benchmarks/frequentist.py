"""Frequentist benchmark estimators for precision/covariance matrices."""

import numpy as np
from sklearn.covariance import GraphicalLassoCV, LedoitWolf


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
    T, p = Y.shape
    Sigma_hat = np.cov(Y, rowvar=False, bias=False)

    if T <= p:
        return Sigma_hat, None

    try:
        Omega_hat = np.linalg.inv(Sigma_hat)
        return Sigma_hat, Omega_hat
    except np.linalg.LinAlgError:
        return Sigma_hat, None


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
    lw = LedoitWolf().fit(Y)
    Sigma_hat = lw.covariance_
    Omega_hat = lw.precision_
    shrinkage_intensity = lw.shrinkage_
    return Sigma_hat, Omega_hat, float(shrinkage_intensity)


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
    kwargs = {"cv": cv, "assume_centered": True}
    if alphas is not None:
        kwargs["alphas"] = alphas

    gl = GraphicalLassoCV(**kwargs).fit(Y)
    Sigma_hat = gl.covariance_
    Omega_hat = gl.precision_
    alpha_selected = gl.alpha_
    return Sigma_hat, Omega_hat, float(alpha_selected)
