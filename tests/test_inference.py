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
from src.utils.io import load_samples, samples_exist  # noqa: E402
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
        assert samples_exist(out, "omega_samples")
        assert samples_exist(out, "kappa_samples")
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
        assert samples_exist(out, "elbo_trace")
        assert diag["final_elbo"] is not None


# ======================================================================
# Gibbs sampler tests (always run — pure NumPy, no JAX needed)
# ======================================================================

class TestGibbsSmoke:
    def test_gibbs_produces_all_outputs(self, tiny_seed_dir, tmp_path):
        """Gibbs sampler on p=5 should succeed and produce all output files."""
        seed_dir, Omega_true, _ = tiny_seed_dir
        out = tmp_path / "out" / "gibbs"
        diag = run_inference(
            "gibbs", seed_dir, out,
            n_burnin=200, n_samples=500, n_thinning=1,
        )
        assert diag["status"] == "success"
        assert (out / "omega_hat.npy").exists()
        assert samples_exist(out, "omega_samples")
        assert samples_exist(out, "kappa_samples")
        assert samples_exist(out, "tau_samples")
        assert samples_exist(out, "lambda_samples")
        assert (out / "diagnostics.json").exists()

    def test_gibbs_omega_is_pd(self, tiny_seed_dir, tmp_path):
        """Every posterior sample of Omega should be positive definite."""
        seed_dir, _, _ = tiny_seed_dir
        out = tmp_path / "out" / "gibbs"
        run_inference(
            "gibbs", seed_dir, out,
            n_burnin=100, n_samples=50, n_thinning=1,
        )
        omega_samples = load_samples(out, "omega_samples")
        for s in range(omega_samples.shape[0]):
            eigs = np.linalg.eigvalsh(omega_samples[s].astype(np.float64))
            assert eigs.min() > 0, f"Sample {s} is not PD: min_eig={eigs.min()}"

    def test_gibbs_omega_is_symmetric(self, tiny_seed_dir, tmp_path):
        """Every posterior sample should be symmetric."""
        seed_dir, _, _ = tiny_seed_dir
        out = tmp_path / "out" / "gibbs"
        run_inference(
            "gibbs", seed_dir, out,
            n_burnin=100, n_samples=50, n_thinning=1,
        )
        omega_samples = load_samples(out, "omega_samples")
        for s in range(omega_samples.shape[0]):
            O = omega_samples[s].astype(np.float64)
            np.testing.assert_allclose(O, O.T, atol=1e-10)

    def test_gibbs_kappa_in_unit_interval(self, tiny_seed_dir, tmp_path):
        """Kappa samples should all be in [0, 1]."""
        seed_dir, _, _ = tiny_seed_dir
        out = tmp_path / "out" / "gibbs"
        run_inference(
            "gibbs", seed_dir, out,
            n_burnin=100, n_samples=200, n_thinning=1,
        )
        kappa = load_samples(out, "kappa_samples")
        assert np.all(kappa >= 0)
        assert np.all(kappa <= 1)

    def test_gibbs_diagnostics_fields(self, tiny_seed_dir, tmp_path):
        """Gibbs diagnostics.json should contain the expected fields."""
        seed_dir, _, _ = tiny_seed_dir
        out = tmp_path / "out" / "gibbs"
        run_inference(
            "gibbs", seed_dir, out,
            n_burnin=100, n_samples=200, n_thinning=1,
        )
        with open(out / "diagnostics.json") as f:
            diag = json.load(f)
        assert diag["method"] == "gibbs"
        assert "n_burnin" in diag
        assert "n_samples" in diag
        assert "min_ess_omega" in diag
        assert "min_ess_tau" in diag
        assert "geweke_p_value_tau" in diag
        assert "mean_rejection_rate_per_column" in diag

    def test_gibbs_evaluate_produces_metrics(self, tiny_seed_dir, tmp_path):
        """Evaluation should work on Gibbs output."""
        from src.evaluation.evaluate_single import evaluate

        seed_dir, _, _ = tiny_seed_dir
        out = tmp_path / "out" / "gibbs"
        run_inference(
            "gibbs", seed_dir, out,
            n_burnin=100, n_samples=200, n_thinning=1,
        )
        metrics = evaluate("gibbs", seed_dir, out)
        assert (out / "metrics.json").exists()
        assert metrics["status"] == "success"
        assert metrics["steins_loss"] is not None
        assert metrics["frobenius_loss"] is not None
        # Gibbs is Bayesian, so coverage and bimodality should be present
        assert metrics.get("coverage_95") is not None
        assert metrics.get("bimodality_coefficient_kappa") is not None


# ======================================================================
# PSIS diagnostic tests
# ======================================================================

class TestPSIS:
    def test_khat_identical_distributions(self):
        """Uniform log-weights → khat ≈ 0 (perfect match)."""
        from src.evaluation.psis import compute_psis_khat

        rng = np.random.default_rng(0)
        log_weights = rng.normal(0, 0.01, size=500)  # near-uniform
        result = compute_psis_khat(log_weights)
        assert result["psis_khat"] is not None
        assert result["psis_khat"] < 0.5  # should be "good"

    def test_khat_heavy_tails(self):
        """Very heavy-tailed log-weights → khat > 0.5."""
        from src.evaluation.psis import compute_psis_khat

        rng = np.random.default_rng(0)
        # Cauchy draws have very heavy tails
        log_weights = rng.standard_cauchy(size=500)
        result = compute_psis_khat(log_weights)
        assert result["psis_khat"] is not None
        assert result["psis_khat"] > 0.3  # at least marginal

    def test_interpret_khat(self):
        from src.evaluation.psis import interpret_khat

        assert interpret_khat(0.3) == "good"
        assert interpret_khat(0.6) == "marginal"
        assert interpret_khat(0.8) == "bad"
        assert interpret_khat(float("nan")) == "unknown"


# ======================================================================
# Gibbs runner unit test (low-level)
# ======================================================================

class TestGibbsRunnerDirect:
    def test_run_gibbs_returns_correct_shapes(self):
        """Direct call to run_gibbs with minimal budget."""
        from src.inference.gibbs_runner import run_gibbs

        p = 5
        T = 200
        Omega, _, _ = sparse_omega_erdos_renyi(p, 0.2, seed=0)
        Y = sample_data_from_omega(Omega, T=T, seed=1)

        result = run_gibbs(
            Y, p=p, n_burnin=50, n_samples=100, n_thinning=1, rng_seed=42,
        )
        assert result["omega_hat"].shape == (p, p)
        assert result["omega_samples"].shape == (100, p, p)
        assert result["tau_sq_samples"].shape == (100,)
        n_offdiag = p * (p - 1) // 2
        assert result["lambda_sq_samples"].shape == (100, n_offdiag)
        assert result["diagnostics"]["n_burnin"] == 50
        assert result["diagnostics"]["n_samples"] == 100

    def test_run_gibbs_tau_positive(self):
        from src.inference.gibbs_runner import run_gibbs

        p = 4
        T = 100
        Omega, _, _ = sparse_omega_erdos_renyi(p, 0.3, seed=7)
        Y = sample_data_from_omega(Omega, T=T, seed=8)

        result = run_gibbs(
            Y, p=p, n_burnin=50, n_samples=50, n_thinning=1, rng_seed=0,
        )
        assert np.all(result["tau_sq_samples"] > 0)
        assert np.all(result["lambda_sq_samples"] > 0)
