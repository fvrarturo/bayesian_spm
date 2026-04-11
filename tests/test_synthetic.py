"""Unit tests for the synthetic data generation pipeline.

These tests correspond to the deliverables in ``_info/WORK1.md`` §8.
Each test class covers one component of the pipeline.  Tests are
deliberately fast (seconds total) so they can run on every commit.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from src.portfolio.gmv import gmv_weights  # noqa: E402
from src.utils.configs import (  # noqa: E402
    N_BLOCKS_MAP,
    compute_configs,
    expected_config_count,
)
from src.utils.matrix_utils import (  # noqa: E402
    assemble_precision_matrix,
    sample_data_from_omega,
    sparse_omega_band,
    sparse_omega_block_diagonal,
    sparse_omega_erdos_renyi,
)
from src.utils.validation import (  # noqa: E402
    validate_data,
    validate_omega,
    validate_sigma,
)


# ----------------------------------------------------------------------
# Erdos-Renyi generator
# ----------------------------------------------------------------------

class TestSparseOmegaErdosRenyi:
    @pytest.mark.parametrize("p", [10, 20, 50])
    def test_is_pd(self, p):
        Omega, _, _ = sparse_omega_erdos_renyi(p, sparsity=0.10, seed=0)
        eigs = np.linalg.eigvalsh(Omega)
        assert eigs.min() > 0
        assert eigs.min() >= 0.1 - 1e-10

    @pytest.mark.parametrize("p", [10, 20, 50])
    def test_is_symmetric(self, p):
        Omega, _, _ = sparse_omega_erdos_renyi(p, sparsity=0.10, seed=0)
        np.testing.assert_allclose(Omega, Omega.T, atol=1e-12)

    def test_sparsity_in_range(self):
        p = 50
        s = 0.10
        n_possible = p * (p - 1) // 2
        # Over a handful of seeds, realized sparsity should concentrate
        # around s.  Use a factor-of-2 window on each side.
        for seed in range(5):
            _, edge_set, _ = sparse_omega_erdos_renyi(p, sparsity=s, seed=seed)
            realized = len(edge_set) / n_possible
            assert 0.5 * s <= realized <= 2.0 * s, f"seed={seed} realized={realized}"

    def test_diagnostics_reports_shift(self):
        """At high density the diagonal shift should be nonzero."""
        _, _, diag = sparse_omega_erdos_renyi(p=30, sparsity=0.6, seed=1)
        assert "diagonal_shift" in diag
        assert "min_eigenvalue_pre_shift" in diag
        assert "n_edges" in diag
        # With 60% sparsity and random signs, PD fix-up is almost guaranteed.
        assert diag["diagonal_shift"] >= 0.0

    def test_edge_set_matches_nonzeros(self):
        """Every edge in edge_set corresponds to a nonzero off-diagonal entry."""
        p = 40
        Omega, edge_set, _ = sparse_omega_erdos_renyi(p, sparsity=0.15, seed=7)
        for (i, j) in edge_set:
            assert abs(Omega[i, j]) > 1e-12
            assert Omega[i, j] == Omega[j, i]

        # And there are no nonzeros outside edge_set.
        idx = np.triu_indices(p, k=1)
        for k in range(len(idx[0])):
            i, j = int(idx[0][k]), int(idx[1][k])
            if (i, j) not in edge_set:
                assert abs(Omega[i, j]) < 1e-12

    def test_reproducible(self):
        """Same seed -> identical output."""
        O1, e1, _ = sparse_omega_erdos_renyi(p=25, sparsity=0.1, seed=123)
        O2, e2, _ = sparse_omega_erdos_renyi(p=25, sparsity=0.1, seed=123)
        np.testing.assert_array_equal(O1, O2)
        assert e1 == e2


# ----------------------------------------------------------------------
# Block-diagonal generator
# ----------------------------------------------------------------------

class TestSparseOmegaBlockDiagonal:
    @pytest.mark.parametrize("p,n_blocks", [(10, 2), (50, 5), (100, 5)])
    def test_is_pd(self, p, n_blocks):
        Omega, _, _ = sparse_omega_block_diagonal(
            p=p, n_blocks=n_blocks, intra_sparsity=0.30, seed=0
        )
        eigs = np.linalg.eigvalsh(Omega)
        assert eigs.min() > 0
        assert eigs.min() >= 0.1 - 1e-10

    @pytest.mark.parametrize("p,n_blocks", [(10, 2), (50, 5), (100, 5)])
    def test_is_symmetric(self, p, n_blocks):
        Omega, _, _ = sparse_omega_block_diagonal(
            p=p, n_blocks=n_blocks, intra_sparsity=0.30, seed=0
        )
        np.testing.assert_allclose(Omega, Omega.T, atol=1e-12)

    @pytest.mark.parametrize("p,n_blocks", [(10, 2), (50, 5), (100, 5)])
    def test_no_inter_block_edges(self, p, n_blocks):
        """Defining invariant of the block-diagonal structure."""
        Omega, edge_set, _ = sparse_omega_block_diagonal(
            p=p, n_blocks=n_blocks, intra_sparsity=0.30, seed=3
        )
        block_size = p // n_blocks
        last_block_start = (n_blocks - 1) * block_size

        def block_of(i: int) -> int:
            if i >= last_block_start:
                return n_blocks - 1
            return i // block_size

        for (i, j) in edge_set:
            assert block_of(i) == block_of(j), f"inter-block edge ({i},{j})"

        # Also verify the matrix itself has zeros off-block.
        for i in range(p):
            for j in range(i + 1, p):
                if block_of(i) != block_of(j):
                    assert abs(Omega[i, j]) < 1e-12


# ----------------------------------------------------------------------
# Band generator (included because matrix_utils still exports it)
# ----------------------------------------------------------------------

class TestSparseOmegaBand:
    @pytest.mark.parametrize("p", [10, 20, 50])
    def test_is_pd(self, p):
        Omega, _, _ = sparse_omega_band(p, bandwidth=2, seed=0)
        eigs = np.linalg.eigvalsh(Omega)
        assert eigs.min() > 0


# ----------------------------------------------------------------------
# Data sampling
# ----------------------------------------------------------------------

class TestDataSampling:
    def test_shape(self):
        Omega, _, _ = sparse_omega_erdos_renyi(p=20, sparsity=0.1, seed=0)
        Y = sample_data_from_omega(Omega, T=100, seed=1)
        assert Y.shape == (100, 20)

    def test_finite(self):
        Omega, _, _ = sparse_omega_erdos_renyi(p=20, sparsity=0.1, seed=0)
        Y = sample_data_from_omega(Omega, T=100, seed=1)
        assert np.isfinite(Y).all()

    def test_sample_cov_rank_T_greater_than_p(self):
        p = 20
        Omega, _, _ = sparse_omega_erdos_renyi(p=p, sparsity=0.1, seed=0)
        Y = sample_data_from_omega(Omega, T=100, seed=2)
        rank = np.linalg.matrix_rank(np.cov(Y, rowvar=False))
        assert rank == p

    def test_sample_cov_rank_T_less_than_p(self):
        p = 20
        Omega, _, _ = sparse_omega_erdos_renyi(p=p, sparsity=0.1, seed=0)
        Y = sample_data_from_omega(Omega, T=15, seed=2)
        rank = np.linalg.matrix_rank(np.cov(Y, rowvar=False))
        # When T <= p, rank is T - 1 (degrees of freedom lost to demeaning).
        assert rank == 14

    def test_reproducible(self):
        Omega, _, _ = sparse_omega_erdos_renyi(p=15, sparsity=0.1, seed=0)
        Y1 = sample_data_from_omega(Omega, T=50, seed=999)
        Y2 = sample_data_from_omega(Omega, T=50, seed=999)
        np.testing.assert_array_equal(Y1, Y2)


# ----------------------------------------------------------------------
# Inversion round-trip
# ----------------------------------------------------------------------

class TestInversion:
    @pytest.mark.parametrize("p", [10, 20, 50])
    def test_reconstruction(self, p):
        Omega, _, _ = sparse_omega_erdos_renyi(p=p, sparsity=0.1, seed=0)
        Sigma = np.linalg.inv(Omega)
        np.testing.assert_allclose(Omega @ Sigma, np.eye(p), atol=1e-8)


# ----------------------------------------------------------------------
# assemble_precision_matrix
# ----------------------------------------------------------------------

class TestAssemblePrecisionMatrix:
    def test_roundtrip(self):
        p = 5
        rng = np.random.default_rng(0)
        offdiag = rng.normal(size=p * (p - 1) // 2)
        diag = rng.uniform(1, 3, size=p)
        Omega = assemble_precision_matrix(offdiag, diag, p)
        assert Omega.shape == (p, p)
        np.testing.assert_allclose(Omega, Omega.T)
        np.testing.assert_allclose(np.diag(Omega), diag)


# ----------------------------------------------------------------------
# Oracle portfolio
# ----------------------------------------------------------------------

class TestOraclePortfolio:
    def test_gmv_weights_sum_to_one(self):
        Omega, _, _ = sparse_omega_erdos_renyi(p=20, sparsity=0.1, seed=0)
        w = gmv_weights(Omega)
        assert abs(w.sum() - 1.0) < 1e-10


# ----------------------------------------------------------------------
# Config manifest
# ----------------------------------------------------------------------

class TestConfigManifest:
    def test_count(self):
        configs = compute_configs()
        assert len(configs) == 84

    def test_count_matches_expected(self):
        assert len(compute_configs()) == expected_config_count()

    def test_no_skipped_configs_generated(self):
        """Skipped (s=0.30, gamma=0.90) combinations should be absent."""
        for c in compute_configs():
            assert not (c["sparsity"] == 0.30 and c["gamma"] == 0.90)

    def test_unique_config_ids(self):
        ids = [c["config_id"] for c in compute_configs()]
        assert ids == list(range(len(ids)))

    def test_T_is_ceil_p_over_gamma(self):
        for c in compute_configs():
            assert c["T"] == math.ceil(c["p"] / c["gamma"])

    def test_block_diagonal_has_n_blocks(self):
        for c in compute_configs():
            if c["graph"] == "block_diagonal":
                assert c["n_blocks"] == N_BLOCKS_MAP[c["p"]]
            else:
                assert c["n_blocks"] is None

    def test_dir_paths_unique(self):
        paths = [c["dir_path"] for c in compute_configs()]
        assert len(paths) == len(set(paths))

    def test_signal_range_matches(self):
        for c in compute_configs():
            assert c["signal_range"] == [0.3, 0.8]


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

class TestValidateOmega:
    def test_accepts_valid(self):
        p = 20
        Omega, edges, diag = sparse_omega_erdos_renyi(p=p, sparsity=0.1, seed=0)
        ok, d = validate_omega(
            Omega, edges, p, diagonal_shift=diag["diagonal_shift"]
        )
        assert ok, d

    def test_rejects_non_symmetric(self):
        p = 10
        Omega = np.eye(p)
        Omega[0, 1] = 0.5  # one-sided asymmetry
        ok, d = validate_omega(Omega, edge_set=set(), p=p)
        assert not ok
        assert "symmetric" in d["error"]

    def test_rejects_not_pd(self):
        p = 5
        Omega = np.eye(p)
        Omega[0, 0] = -1.0
        ok, d = validate_omega(Omega, edge_set=set(), p=p)
        assert not ok
        assert "PD" in d["error"] or "eig" in d["error"].lower()

    def test_rejects_sparsity_pattern_mismatch(self):
        p = 10
        Omega, edges, _ = sparse_omega_erdos_renyi(p=p, sparsity=0.2, seed=0)
        # Forge an edge set missing one entry.
        bad = set(edges)
        bad.pop()
        ok, d = validate_omega(Omega, edge_set=bad, p=p)
        assert not ok
        assert "sparsity pattern" in d["error"]

    def test_warns_on_large_diagonal_shift(self):
        p = 10
        Omega, edges, _ = sparse_omega_erdos_renyi(p=p, sparsity=0.1, seed=0)
        ok, d = validate_omega(Omega, edges, p, diagonal_shift=5.0)
        assert ok
        assert any("diagonal shift" in w for w in d["warnings"])


class TestValidateSigma:
    def test_accepts_valid_inverse(self):
        p = 20
        Omega, _, _ = sparse_omega_erdos_renyi(p=p, sparsity=0.1, seed=0)
        Sigma = np.linalg.inv(Omega)
        ok, _ = validate_sigma(Sigma, Omega, p)
        assert ok

    def test_rejects_bad_inverse(self):
        p = 10
        Omega = np.eye(p)
        Sigma = 2 * np.eye(p)  # not actually Omega^{-1}
        ok, d = validate_sigma(Sigma, Omega, p)
        assert not ok
        assert "reconstruction_max_err" in d


class TestValidateData:
    def test_accepts_valid(self):
        Omega, _, _ = sparse_omega_erdos_renyi(p=20, sparsity=0.1, seed=0)
        Sigma = np.linalg.inv(Omega)
        Y = sample_data_from_omega(Omega, T=200, seed=1)
        ok, _ = validate_data(Y, T=200, p=20, Sigma=Sigma)
        assert ok

    def test_rejects_wrong_shape(self):
        Y = np.zeros((10, 5))
        ok, d = validate_data(Y, T=20, p=5)
        assert not ok

    def test_rejects_nan(self):
        Y = np.zeros((30, 5))
        Y[0, 0] = np.nan
        ok, d = validate_data(Y, T=30, p=5)
        assert not ok

    def test_rejects_inf(self):
        Y = np.zeros((30, 5))
        Y[0, 0] = np.inf
        ok, d = validate_data(Y, T=30, p=5)
        assert not ok


# ----------------------------------------------------------------------
# End-to-end: generate_single_config
# ----------------------------------------------------------------------

class TestGenerateSingleConfig:
    def test_produces_all_files(self, tmp_path):
        from generate_synthetic_data import generate_single_config

        config = {
            "config_id": 0,
            "p": 20,
            "T": 50,
            "gamma": 0.40,
            "graph": "erdos_renyi",
            "sparsity": 0.10,
            "n_blocks": None,
            "signal_range": [0.3, 0.8],
            "dir_path": "erdos_renyi/p020/gamma040/s010",
            "n_seeds": 1,
        }
        ok, info = generate_single_config(config, seed=0, output_base_dir=tmp_path)
        assert ok, info
        seed_dir = tmp_path / config["dir_path"] / "seed_00"
        for fname in ("omega_true.npy", "sigma_true.npy", "Y.npy", "metadata.json"):
            assert (seed_dir / fname).exists(), fname

        # Verify metadata roundtrip
        with open(seed_dir / "metadata.json") as f:
            md = json.load(f)
        assert md["config_id"] == 0
        assert md["p"] == 20
        assert md["T"] == 50
        assert md["graph"] == "erdos_renyi"
        assert md["seed"] == 0
        assert md["graph_seed"] == 0
        assert md["data_seed"] == 10000
        assert "edge_set" in md
        assert "warnings" in md
        assert md["n_edges"] == len(md["edge_set"])

        # Verify arrays load and have the right shapes.
        Omega = np.load(seed_dir / "omega_true.npy")
        Sigma = np.load(seed_dir / "sigma_true.npy")
        Y = np.load(seed_dir / "Y.npy")
        assert Omega.shape == (20, 20)
        assert Sigma.shape == (20, 20)
        assert Y.shape == (50, 20)
        np.testing.assert_allclose(Omega @ Sigma, np.eye(20), atol=1e-8)

    def test_block_diagonal(self, tmp_path):
        from generate_synthetic_data import generate_single_config

        config = {
            "config_id": 1,
            "p": 50,
            "T": 120,
            "gamma": 0.42,
            "graph": "block_diagonal",
            "sparsity": 0.30,
            "n_blocks": 5,
            "signal_range": [0.3, 0.8],
            "dir_path": "block_diagonal/p050/gamma042/s030",
            "n_seeds": 1,
        }
        ok, info = generate_single_config(config, seed=0, output_base_dir=tmp_path)
        assert ok, info
        assert info["p"] == 50
        assert info["T"] == 120

    def test_seed_independence(self, tmp_path):
        """Different seeds should yield different edge sets."""
        from generate_synthetic_data import generate_single_config

        config = {
            "config_id": 2,
            "p": 30,
            "T": 100,
            "gamma": 0.30,
            "graph": "erdos_renyi",
            "sparsity": 0.10,
            "n_blocks": None,
            "signal_range": [0.3, 0.8],
            "dir_path": "erdos_renyi/p030/gamma030/s010",
            "n_seeds": 2,
        }
        ok0, info0 = generate_single_config(config, seed=0, output_base_dir=tmp_path)
        ok1, info1 = generate_single_config(config, seed=1, output_base_dir=tmp_path)
        assert ok0 and ok1
        assert info0["edge_set"] != info1["edge_set"]
        assert info0["data_seed"] != info1["data_seed"]


class TestAudit:
    def test_audit_detects_missing(self, tmp_path):
        """Audit should flag a seed directory that was never created."""
        from audit_synthetic_data import audit
        from generate_synthetic_data import generate_single_config

        manifest = [
            {
                "config_id": 0,
                "p": 20,
                "T": 50,
                "gamma": 0.4,
                "graph": "erdos_renyi",
                "sparsity": 0.1,
                "n_blocks": None,
                "signal_range": [0.3, 0.8],
                "n_seeds": 2,
                "dir_path": "erdos_renyi/p020/gamma040/s010",
            }
        ]
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Generate only seed 0; seed 1 will be missing.
        generate_single_config(manifest[0], seed=0, output_base_dir=tmp_path)

        summary = audit(manifest_path, tmp_path, strict=True)
        assert summary["n_seeds_total"] == 2
        assert summary["status_counts"]["ok"] == 1
        assert summary["status_counts"]["missing_dir"] == 1
        assert summary["n_failures"] == 1
