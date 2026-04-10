"""Utilities for precision matrix assembly and synthetic data generation."""

import numpy as np
import networkx as nx


def assemble_precision_matrix(omega_offdiag, omega_diag, p):
    """Assemble a symmetric precision matrix from off-diagonal and diagonal vectors.

    Parameters
    ----------
    omega_offdiag : np.ndarray, shape (p*(p-1)/2,)
        Upper-triangular off-diagonal elements.
    omega_diag : np.ndarray, shape (p,)
        Diagonal elements.
    p : int
        Dimension.

    Returns
    -------
    Omega : np.ndarray, shape (p, p)
    """
    Omega = np.zeros((p, p))
    idx_upper = np.triu_indices(p, k=1)
    Omega[idx_upper] = omega_offdiag
    Omega = Omega + Omega.T
    np.fill_diagonal(Omega, omega_diag)
    return Omega


def _graph_to_omega(G, p, signal_range, rng):
    """Convert a networkx graph to a sparse PD precision matrix.

    Assigns random edge weights, symmetrises, and shifts the diagonal
    to guarantee positive definiteness.
    """
    Omega = np.eye(p)
    edge_set = set()
    lo, hi = signal_range

    for i, j in G.edges():
        val = rng.choice([-1, 1]) * rng.uniform(lo, hi)
        Omega[i, j] = val
        Omega[j, i] = val
        edge_set.add((min(i, j), max(i, j)))

    eigmin = np.linalg.eigvalsh(Omega).min()
    if eigmin < 0.1:
        Omega += (0.1 - eigmin) * np.eye(p)

    return Omega, edge_set


def sparse_omega_erdos_renyi(p, sparsity=0.10, signal_range=(0.3, 0.8), seed=42):
    """Generate a sparse PD precision matrix with Erdos-Renyi random graph structure.

    Parameters
    ----------
    p : int
        Dimension.
    sparsity : float
        Edge probability.
    signal_range : tuple
        (min, max) absolute value of nonzero off-diagonal entries.
    seed : int

    Returns
    -------
    Omega : np.ndarray, shape (p, p)
        Positive definite precision matrix.
    edge_set : set of (i, j) tuples
        True nonzero off-diagonal positions.
    """
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(p, sparsity, seed=int(rng.integers(1e9)))
    return _graph_to_omega(G, p, signal_range, rng)


def sparse_omega_band(p, bandwidth=2, signal_range=(0.3, 0.8), seed=42):
    """Generate a banded sparse PD precision matrix.

    Parameters
    ----------
    p : int
    bandwidth : int
    signal_range : tuple
    seed : int

    Returns
    -------
    Omega : np.ndarray, shape (p, p)
    edge_set : set of (i, j) tuples
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(p))
    for i in range(p):
        for j in range(i + 1, min(i + bandwidth + 1, p)):
            G.add_edge(i, j)
    return _graph_to_omega(G, p, signal_range, rng)


def sparse_omega_block(p, n_blocks=5, intra_sparsity=0.3, signal_range=(0.3, 0.8), seed=42):
    """Generate a block-diagonal sparse PD precision matrix.

    Parameters
    ----------
    p : int
    n_blocks : int
    intra_sparsity : float
    signal_range : tuple
    seed : int

    Returns
    -------
    Omega : np.ndarray, shape (p, p)
    edge_set : set of (i, j) tuples
    """
    rng = np.random.default_rng(seed)
    block_size = p // n_blocks
    G = nx.Graph()
    G.add_nodes_from(range(p))

    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else p
        for i in range(start, end):
            for j in range(i + 1, end):
                if rng.random() < intra_sparsity:
                    G.add_edge(i, j)

    return _graph_to_omega(G, p, signal_range, rng)


def sample_data_from_omega(Omega, T, seed=42):
    """Sample iid Gaussian observations from a given precision matrix.

    Parameters
    ----------
    Omega : np.ndarray, shape (p, p)
        Positive definite precision matrix.
    T : int
        Number of samples.
    seed : int

    Returns
    -------
    Y : np.ndarray, shape (T, p)
        Zero-mean observation matrix.
    """
    rng = np.random.default_rng(seed)
    Sigma = np.linalg.inv(Omega)
    Y = rng.multivariate_normal(np.zeros(Omega.shape[0]), Sigma, size=T)
    return Y
