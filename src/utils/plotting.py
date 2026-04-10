"""Plotting utilities for the graphical horseshoe project."""

import numpy as np


def plot_precision_heatmap(Omega, title="Precision Matrix", ax=None):
    """Plot a heatmap of the precision matrix.

    Parameters
    ----------
    Omega : np.ndarray, shape (p, p)
    title : str
    ax : matplotlib Axes or None
    """
    raise NotImplementedError


def plot_shrinkage_profile(kappa_nuts, kappa_advi, ax=None):
    """Plot side-by-side shrinkage coefficient distributions for NUTS vs ADVI.

    Parameters
    ----------
    kappa_nuts : np.ndarray
        Shrinkage coefficients from NUTS posterior.
    kappa_advi : np.ndarray
        Shrinkage coefficients from ADVI posterior.
    ax : matplotlib Axes or None
    """
    raise NotImplementedError


def plot_eigenvalue_comparison(eigenvalues_dict, true_eigenvalues=None, ax=None):
    """Plot sorted eigenvalues for multiple estimators.

    Parameters
    ----------
    eigenvalues_dict : dict
        {method_name: eigenvalues_array}.
    true_eigenvalues : np.ndarray or None
    ax : matplotlib Axes or None
    """
    raise NotImplementedError


def plot_elbo_trace(losses, title="ELBO Trace", ax=None):
    """Plot ELBO convergence curve from ADVI.

    Parameters
    ----------
    losses : np.ndarray
        ELBO values per iteration.
    title : str
    ax : matplotlib Axes or None
    """
    raise NotImplementedError
