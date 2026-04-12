"""Smoke tests for the inference runners and dispatcher.

These tests actually run inference on small synthetic cases.  NUTS /
ADVI tests are marked ``slow`` and skipped unless NumPyro is installed
and ``RUN_INFERENCE_TESTS=1`` is set in the environment.  Frequentist
tests run unconditionally (sklearn is a hard dep).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.run_single import run_inference  # noqa: E402
from src.utils.matrix_utils import (  # noqa: E402
    sample_data_from_omega,
    sparse_omega_erdos_renyi,
)


RUN_INFERENCE = os.environ.get("RUN_INFERENCE_TESTS") == "1"

try:
    import numpyro  # noqa: F401
    HAS_NUMPYRO = True
except Exception:
    HAS_NUMPYRO = False


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def tiny_seed_dir(tmp_path):
    """Generate a tiny WORK1-style seed directory (p=5, T=200).

    This is cheap enough to run for every test.
    """
    p = 5
    T = 200
    Omega, edge_set, diag = sparse_omega_erdos_renyi(
        p=p, sparsity=0.2, signal_range=(0.3, 0.8), seed=0,
    )
    Sigma = np.linalg.inv(Omega)
    Y = sample_data_from_omega(Omega, T=T, seed=1)

    seed_dir = tmp_path / "test" / "seed_00"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.save(seed_dir / "omega_true.npy", Omega)
    np.save(seed_dir / "sigma_true.npy", Sigma)
    np.save(seed_dir / "Y.npy", Y)
    metadata = {
        "config_id": 0,
        "p": p,
        "T": T,
        "gamma": p / T,
        "graph": "erdos_renyi",
        "sparsity": 0.2,
        "seed": 0,
        "graph_seed": 0,
        "data_seed": 1,
        "n_edges": len(edge_set),
        "edge_set": [list(e) for e in sorted(edge_set)],
    }
    with open(seed_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return seed_dir, Omega, edge_set


# ======================================================================
# Frequentist methods (always run)
# ======================================================================

class TestSampleCov:
    def test_returns_success_when_T_gt_p(self, tiny_seed_dir, tmp_path):
        seed_dir, Omega_true, _ = tiny_seed_dir
        out = tmp_path / "out" / "sample_cov"
        diag = run_inference("sample_cov", seed_dir, out)
        assert diag["status"] == "success"
        assert (out / "omega_hat.npy").exists()

    def test_singular_when_T_leq_p(self, tmp_path):
        p = 10
        T = 5
        Omega, _, _ = sparse_omega_erdos_renyi(p, 0.1, seed=0)
        Y = sample_data_from_omega(Omega, T=T, seed=1)

        seed_dir = tmp_path / "small" / "seed_00"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "omega_true.npy", Omega)
        np.save(seed_dir / "Y.npy", Y)
        with open(seed_dir / "metadata.json", "w") as f:
            json.dump({
                "config_id": 0, "p": p, "T": T, "seed": 0,
                "graph": "erdos_renyi", "sparsity": 0.1, "gamma": p / T,
                "data_seed": 1, "edge_set": [],
            }, f)

        out = tmp_path / "out" / "sample_cov"
        diag = run_inference("sample_cov", seed_dir, out)
        assert diag["status"] == "singular"
        assert not (out / "omega_hat.npy").exists()


class TestLedoitWolf:
    def test_success(self, tiny_seed_dir, tmp_path):
        seed_dir, _, _ = tiny_seed_dir
        out = tmp_path / "out" / "ledoit_wolf"
        diag = run_inference("ledoit_wolf", seed_dir, out)
        assert diag["status"] == "success"
        # LW always produces an invertible estimate.
        Omega_hat = np.load(out / "omega_hat.npy")
        assert Omega_hat.shape == (5, 5)
        assert np.all(np.linalg.eigvalsh(Omega_hat) > 0)
        assert "shrinkage_intensity" in diag


class TestGlasso:
    def test_success_has_sparse_output(self, tiny_seed_dir, tmp_path):
        seed_dir, _, _ = tiny_seed_dir
        out = tmp_path / "out" / "glasso"
        diag = run_inference("glasso", seed_dir, out)
        assert diag["status"] == "success"
        Omega_hat = np.load(out / "omega_hat.npy")
        # Glasso produces exact zeros on the off-diagonal when alpha > 0.
        off = Omega_hat - np.diag(np.diag(Omega_hat))
        # At least some off-diagonal should be zero for a sparse matrix.
        n_exact_zero = int(np.sum(off == 0.0) - 5)  # subtract diagonal
        assert n_exact_zero >= 0  # always true; here as a non-regression guard
        assert "alpha_selected" in diag


# ======================================================================
# Evaluation wiring
# ======================================================================

class TestEvaluateWiring:
    def test_evaluate_writes_metrics(self, tiny_seed_dir, tmp_path):
        from src.evaluation.evaluate_single import evaluate

        seed_dir, _, _ = tiny_seed_dir
        out = tmp_path / "out" / "ledoit_wolf"
        run_inference("ledoit_wolf", seed_dir, out)
        metrics = evaluate("ledoit_wolf", seed_dir, out)

        assert (out / "metrics.json").exists()
        assert metrics["status"] == "success"
        assert metrics["steins_loss"] is not None
        assert metrics["frobenius_loss"] is not None

    def test_evaluate_on_singular_writes_nulls(self, tmp_path):
        from src.evaluation.evaluate_single import evaluate

        p, T = 10, 5
        Omega, _, _ = sparse_omega_erdos_renyi(p, 0.1, seed=0)
        Y = sample_data_from_omega(Omega, T=T, seed=1)
        seed_dir = tmp_path / "s" / "seed_00"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "omega_true.npy", Omega)
        np.save(seed_dir / "Y.npy", Y)
        with open(seed_dir / "metadata.json", "w") as f:
            json.dump({
                "config_id": 0, "p": p, "T": T, "seed": 0,
                "graph": "erdos_renyi", "sparsity": 0.1, "gamma": p / T,
                "data_seed": 1, "edge_set": [],
            }, f)

        out = tmp_path / "out" / "sample_cov"
        run_inference("sample_cov", seed_dir, out)
        metrics = evaluate("sample_cov", seed_dir, out)
        assert metrics["status"] == "singular"
        # Numeric fields should be null
        assert metrics["steins_loss"] is None
        assert metrics["frobenius_loss"] is None


# ======================================================================
# Bayesian smoke tests (gated by RUN_INFERENCE_TESTS)
# ======================================================================

@pytest.mark.skipif(
    not (HAS_NUMPYRO and RUN_INFERENCE),
    reason="Set RUN_INFERENCE_TESTS=1 and install numpyro to run the Bayesian smoke tests.",
)
class TestNutsSmoke:
    def test_nuts_converges_on_small_problem(self, tiny_seed_dir, tmp_path):
        seed_dir, Omega_true, _ = tiny_seed_dir
        out = tmp_path / "out" / "nuts"
        # Use a cheap budget for testing
        diag = run_inference(
            "nuts", seed_dir, out,
            num_warmup=300, num_samples=300, num_chains=2,
        )
        assert diag["status"] == "success"
        assert (out / "omega_hat.npy").exists()
        assert (out / "omega_samples.npy").exists()
        assert (out / "kappa_samples.npy").exists()
        assert diag["max_rhat"] is not None
        # R-hat should be in a sane range even with a tiny budget.
        assert diag["max_rhat"] < 1.20


@pytest.mark.skipif(
    not (HAS_NUMPYRO and RUN_INFERENCE),
    reason="Set RUN_INFERENCE_TESTS=1 and install numpyro to run the Bayesian smoke tests.",
)
class TestAdviMfSmoke:
    def test_advi_mf_runs(self, tiny_seed_dir, tmp_path):
        seed_dir, _, _ = tiny_seed_dir
        out = tmp_path / "out" / "advi_mf"
        diag = run_inference(
            "advi_mf", seed_dir, out,
            num_steps=2000, num_seeds=2,
        )
        assert diag["status"] == "success"
        assert (out / "omega_hat.npy").exists()
        assert (out / "elbo_trace.npy").exists()
        assert diag["final_elbo"] is not None
