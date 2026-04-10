"""Tests for synthetic data generation and matrix utilities."""

import numpy as np
import pytest

from src.utils.matrix_utils import (
    assemble_precision_matrix,
    sparse_omega_band,
    sparse_omega_block,
    sparse_omega_erdos_renyi,
)


class TestSparseOmegaGenerators:
    """Verify that generated precision matrices are valid."""

    @pytest.mark.parametrize("p", [10, 20, 50])
    def test_erdos_renyi_is_pd(self, p):
        Omega, _ = sparse_omega_erdos_renyi(p, sparsity=0.10)
        eigenvalues = np.linalg.eigvalsh(Omega)
        assert np.all(eigenvalues > 0), "Omega must be positive definite"

    @pytest.mark.parametrize("p", [10, 20, 50])
    def test_erdos_renyi_is_symmetric(self, p):
        Omega, _ = sparse_omega_erdos_renyi(p, sparsity=0.10)
        np.testing.assert_array_almost_equal(Omega, Omega.T)

    @pytest.mark.parametrize("p", [10, 20, 50])
    def test_band_is_pd(self, p):
        Omega, _ = sparse_omega_band(p, bandwidth=2)
        eigenvalues = np.linalg.eigvalsh(Omega)
        assert np.all(eigenvalues > 0)

    @pytest.mark.parametrize("p", [20, 50])
    def test_block_is_pd(self, p):
        Omega, _ = sparse_omega_block(p, n_blocks=5)
        eigenvalues = np.linalg.eigvalsh(Omega)
        assert np.all(eigenvalues > 0)

    def test_erdos_renyi_sparsity(self):
        p = 50
        Omega, edge_set = sparse_omega_erdos_renyi(p, sparsity=0.10, seed=0)
        max_edges = p * (p - 1) // 2
        observed_sparsity = len(edge_set) / max_edges
        assert 0.01 < observed_sparsity < 0.30, "Sparsity should be roughly around 0.10"
