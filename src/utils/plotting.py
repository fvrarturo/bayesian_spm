"""Plotting utilities for the graphical horseshoe project."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_precision_heatmap(Omega, title="Precision Matrix", ax=None, vmax=None):
    """Plot a heatmap of the precision matrix.

    Parameters
    ----------
    Omega : np.ndarray, shape (p, p)
    title : str
    ax : matplotlib Axes or None
    vmax : float or None
        Symmetric color range [-vmax, vmax]. Auto-detected if None.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    if vmax is None:
        vmax = np.max(np.abs(Omega - np.diag(np.diag(Omega))))
    sns.heatmap(
        Omega,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cmap="RdBu_r",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    return ax


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
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax[0].hist(kappa_nuts, bins=50, density=True, alpha=0.8, color="steelblue")
    ax[0].set_xlabel(r"$\hat{\kappa}_{ij}$")
    ax[0].set_ylabel("Density")
    ax[0].set_title("NUTS")
    ax[0].set_xlim(0, 1)

    ax[1].hist(kappa_advi, bins=50, density=True, alpha=0.8, color="coral")
    ax[1].set_xlabel(r"$\hat{\kappa}_{ij}$")
    ax[1].set_title("ADVI")
    ax[1].set_xlim(0, 1)

    return ax


def plot_eigenvalue_comparison(eigenvalues_dict, true_eigenvalues=None, ax=None):
    """Plot sorted eigenvalues for multiple estimators.

    Parameters
    ----------
    eigenvalues_dict : dict
        {method_name: eigenvalues_array}.
    true_eigenvalues : np.ndarray or None
    ax : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if true_eigenvalues is not None:
        eigs = np.sort(true_eigenvalues)[::-1]
        ax.plot(eigs, "k-", linewidth=2, label="True")

    for name, eigs in eigenvalues_dict.items():
        eigs_sorted = np.sort(np.array(eigs))[::-1]
        ax.plot(eigs_sorted, "--", linewidth=1.5, label=name)

    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue Spectrum Comparison")
    ax.legend()
    return ax


def plot_elbo_trace(losses, title="ELBO Trace", ax=None):
    """Plot ELBO convergence curve from ADVI.

    Parameters
    ----------
    losses : np.ndarray
        Loss values per iteration (negative ELBO).
    title : str
    ax : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    losses = np.array(losses)
    ax.plot(-losses, linewidth=0.5, alpha=0.6, color="gray")

    # Smoothed trace
    window = min(500, len(losses) // 10)
    if window > 1:
        smoothed = np.convolve(-losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, window - 1 + len(smoothed)), smoothed,
                linewidth=1.5, color="steelblue", label=f"Smoothed (w={window})")
        ax.legend()

    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO")
    ax.set_title(title)
    return ax


def plot_posterior_comparison(omega_nuts, omega_advi, omega_true=None,
                              entry_labels=None, ax=None):
    """Plot posterior distributions of selected omega entries from NUTS vs ADVI.

    Parameters
    ----------
    omega_nuts : np.ndarray, shape (n_entries, n_samples)
        NUTS posterior samples for selected entries.
    omega_advi : np.ndarray, shape (n_entries, n_samples)
        ADVI posterior samples for selected entries.
    omega_true : np.ndarray or None, shape (n_entries,)
        True values (if synthetic).
    entry_labels : list of str or None
    ax : array of matplotlib Axes or None
    """
    n_entries = omega_nuts.shape[0]
    if ax is None:
        fig, ax = plt.subplots(1, n_entries, figsize=(4 * n_entries, 3.5))
    if n_entries == 1:
        ax = [ax]

    for k in range(n_entries):
        ax[k].hist(omega_nuts[k], bins=40, density=True, alpha=0.6,
                    color="steelblue", label="NUTS")
        ax[k].hist(omega_advi[k], bins=40, density=True, alpha=0.6,
                    color="coral", label="ADVI")
        if omega_true is not None:
            ax[k].axvline(omega_true[k], color="black", linestyle="--",
                          linewidth=1.5, label="True")
        label = entry_labels[k] if entry_labels else f"Entry {k}"
        ax[k].set_title(label)
        ax[k].legend(fontsize=8)

    return ax
