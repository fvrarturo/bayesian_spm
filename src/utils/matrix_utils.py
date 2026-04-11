"""Utilities for precision matrix assembly and synthetic data generation.

All sparse-Omega generators return a 3-tuple ``(Omega, edge_set, diagnostics)``
where ``diagnostics`` is a dict containing:

- ``diagonal_shift`` : float
    The amount added to the diagonal to enforce positive definiteness.
    Zero if the pre-shift matrix was already PD with min-eig >= 0.1.
- ``min_eigenvalue_pre_shift`` : float
    Minimum eigenvalue BEFORE the diagonal shift was applied.  Useful to
    detect cases where the random edge weights almost broke PD-ness.
- ``n_edges`` : int
    Number of edges actually placed in the graph.
"""

import numpy as np
import networkx as nx


# ----------------------------------------------------------------------
# Low-level assembly
# ----------------------------------------------------------------------

def assemble_precision_matrix(omega_offdiag, omega_diag, p):
    """Assemble a symmetric precision matrix from off-diagonal and diagonal vectors.

    Parameters
    ----------
    omega_offdiag : np.ndarray, shape (p*(p-1)/2,)
        Upper-triangular off-diagonal elements in row-major order, i.e.
        ``(0,1), (0,2), ..., (0,p-1), (1,2), ..., (p-2,p-1)``.
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


# ----------------------------------------------------------------------
# Private: convert a NetworkX graph to a PD precision matrix
# ----------------------------------------------------------------------

def _graph_to_omega(G, p, signal_range, rng):
    """Convert a NetworkX graph to a sparse PD precision matrix.

    Starts from the identity, assigns random signed edge weights from
    ``Uniform(signal_range) * {-1,+1}`` to each graph edge, then shifts
    the diagonal by the smallest amount needed to guarantee
    ``min_eig(Omega) >= 0.1``.

    Returns
    -------
    Omega : np.ndarray, shape (p, p)
    edge_set : set of (i, j) tuples with i < j
    diagnostics : dict
        Keys: ``diagonal_shift``, ``min_eigenvalue_pre_shift``, ``n_edges``.
    """
    Omega = np.eye(p)
    edge_set = set()
    lo, hi = signal_range

    for i, j in G.edges():
        val = rng.choice([-1, 1]) * rng.uniform(lo, hi)
        ii, jj = int(min(i, j)), int(max(i, j))
        Omega[ii, jj] = val
        Omega[jj, ii] = val
        edge_set.add((ii, jj))

    eigmin_pre = float(np.linalg.eigvalsh(Omega).min())
    shift = 0.0
    if eigmin_pre < 0.1:
        shift = 0.1 - eigmin_pre
        Omega = Omega + shift * np.eye(p)

    diagnostics = {
        "diagonal_shift": float(shift),
        "min_eigenvalue_pre_shift": eigmin_pre,
        "n_edges": len(edge_set),
    }
    return Omega, edge_set, diagnostics


# ----------------------------------------------------------------------
# Public graph generators
# ----------------------------------------------------------------------

def sparse_omega_erdos_renyi(p, sparsity=0.10, signal_range=(0.3, 0.8), seed=42):
    """Generate a sparse PD precision matrix with Erdos-Renyi random graph structure.

    Each of the ``C(p, 2)`` possible edges is included independently with
    probability ``sparsity``.

    Parameters
    ----------
    p : int
    sparsity : float
        Edge probability.
    signal_range : tuple(float, float)
        (min, max) absolute value of nonzero off-diagonal entries.
    seed : int

    Returns
    -------
    Omega : np.ndarray, shape (p, p)
    edge_set : set of (i, j) tuples
    diagnostics : dict
    """
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(p, sparsity, seed=int(rng.integers(1_000_000_000)))
    return _graph_to_omega(G, p, signal_range, rng)


def sparse_omega_band(p, bandwidth=2, signal_range=(0.3, 0.8), seed=42):
    """Generate a banded sparse PD precision matrix.

    An edge ``(i, j)`` is present iff ``|i - j| <= bandwidth``.

    Returns
    -------
    Omega, edge_set, diagnostics
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(p))
    for i in range(p):
        for j in range(i + 1, min(i + bandwidth + 1, p)):
            G.add_edge(i, j)
    return _graph_to_omega(G, p, signal_range, rng)


def sparse_omega_block_diagonal(
    p,
    n_blocks=5,
    intra_sparsity=0.30,
    signal_range=(0.3, 0.8),
    seed=42,
):
    """Generate a block-diagonal sparse PD precision matrix.

    The ``p`` nodes are partitioned into ``n_blocks`` contiguous blocks
    of (approximately) equal size.  Edges are placed independently within
    each block with probability ``intra_sparsity``; NO edges are ever
    placed across blocks.  When ``p`` is not divisible by ``n_blocks``,
    the final block absorbs the remainder.

    This mirrors the ``sector structure'' in financial applications,
    where stocks within the same industry are conditionally dependent
    but stocks across industries are (approximately) conditionally
    independent after controlling for the rest of the market.

    Parameters
    ----------
    p : int
    n_blocks : int
    intra_sparsity : float
        Probability of placing an edge between any pair of nodes within
        the same block.  Note this controls LOCAL density: the overall
        (global) sparsity is lower because inter-block pairs are
        structurally zero.
    signal_range : tuple(float, float)
    seed : int

    Returns
    -------
    Omega, edge_set, diagnostics
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


# ----------------------------------------------------------------------
# Data sampling
# ----------------------------------------------------------------------

def sample_data_from_omega(Omega, T, seed=42):
    """Sample iid Gaussian observations from N(0, Omega^{-1}).

    Parameters
    ----------
    Omega : np.ndarray, shape (p, p)
        Positive definite precision matrix.
    T : int
        Number of samples.
    seed : int

    Returns
    -------
    Y : np.ndarray, shape (T, p), float64
    """
    rng = np.random.default_rng(seed)
    Sigma = np.linalg.inv(Omega)
    p = Omega.shape[0]
    Y = rng.multivariate_normal(np.zeros(p), Sigma, size=T)
    return Y
