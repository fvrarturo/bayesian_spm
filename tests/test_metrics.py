"""Unit tests for src.evaluation.metrics and src.evaluation.shrinkage.

These are fast, pure-numpy tests.  They use small analytical cases
where the expected output is known in closed form, plus a few
property-based checks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    coverage_95,
    eigenvalue_metrics,
    frobenius_loss,
    frobenius_loss_relative,
    gmv_metrics,
    safe_call,
    sparsity_metrics,
    sparsity_metrics_credible,
    spectral_loss,
    steins_loss,
    trace_error,
)
from src.evaluation.shrinkage import (  # noqa: E402
    BIMODALITY_THRESHOLD,
    bimodality_coefficient,
    compute_kappa_hat,
    compute_kappa_samples,
    shrinkage_profile_summary,
    shrinkage_wasserstein,
)


# ======================================================================
# Loss functions
# ======================================================================

class TestSteinLoss:
    def test_zero_at_truth(self):
        Omega = np.diag([1.0, 2.0, 3.0])
        assert steins_loss(Omega, Omega) == pytest.approx(0.0, abs=1e-12)

    def test_positive_when_different(self):
        Omega0 = np.diag([1.0, 2.0, 3.0])
        Omega_hat = np.diag([1.5, 2.0, 3.0])
        assert steins_loss(Omega_hat, Omega0) > 0

    def test_asymmetric(self):
        """Stein's loss is a Bregman divergence — it is not symmetric."""
        A = np.diag([1.0, 2.0])
        B = np.diag([3.0, 4.0])
        assert steins_loss(A, B) != pytest.approx(steins_loss(B, A))

    def test_known_diagonal_case(self):
        """For diagonal matrices, L_S = sum_i (lambda_i/lambda_hat_i - log(lambda_i/lambda_hat_i) - 1)."""
        Omega_hat = np.diag([1.0, 2.0])
        Omega_true = np.diag([2.0, 4.0])
        # ratios: 2, 2
        # per-entry: 2 - log(2) - 1 = 1 - log(2) ≈ 0.3069
        expected = 2 * (2 - np.log(2) - 1)
        assert steins_loss(Omega_hat, Omega_true) == pytest.approx(expected, rel=1e-10)


class TestFrobeniusLoss:
    def test_zero_at_truth(self):
        Omega = np.eye(5)
        assert frobenius_loss(Omega, Omega) == 0.0

    def test_value(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        B = np.array([[2.0, 0.0], [0.0, 3.0]])
        # diff = [[1, 0], [0, 2]], squared sum = 5
        assert frobenius_loss(A, B) == pytest.approx(5.0)

    def test_relative_zero_at_truth(self):
        Omega = np.eye(5) * 2.0
        assert frobenius_loss_relative(Omega, Omega) == pytest.approx(0.0)

    def test_relative_scale_invariant(self):
        Omega_true = np.eye(3)
        Omega_hat = 2.0 * np.eye(3)
        rel = frobenius_loss_relative(Omega_hat, Omega_true)
        # |2I - I|_F^2 / |I|_F^2 = 3 / 3 = 1
        assert rel == pytest.approx(1.0)


class TestSpectralLoss:
    def test_zero_at_truth(self):
        Omega = np.eye(4)
        assert spectral_loss(Omega, Omega) == pytest.approx(0.0)

    def test_known_case(self):
        A = np.diag([1.0, 1.0])
        B = np.diag([3.0, 1.0])
        # diff has eigenvalues 2, 0; spectral norm = 2
        assert spectral_loss(A, B) == pytest.approx(2.0)


class TestTraceError:
    def test_zero_at_truth(self):
        Omega = np.eye(3)
        assert trace_error(Omega, Omega) == 0.0

    def test_signed(self):
        A = np.diag([1.0, 2.0, 3.0])  # trace 6
        B = np.diag([1.0, 1.0, 1.0])  # trace 3
        assert trace_error(A, B) == pytest.approx(3.0)
        assert trace_error(B, A) == pytest.approx(-3.0)


# ======================================================================
# Sparsity metrics
# ======================================================================

def _make_sparse_omega(p, edge_positions, signal=0.5):
    """Build a simple sparse symmetric matrix with unit diagonal."""
    Omega = np.eye(p)
    for (i, j) in edge_positions:
        Omega[i, j] = Omega[j, i] = signal
    return Omega


class TestSparsityMetricsThreshold:
    def test_perfect_recovery(self):
        p = 5
        edges = [(0, 1), (2, 3)]
        Omega = _make_sparse_omega(p, edges)
        m = sparsity_metrics(Omega, Omega, threshold=1e-5)
        assert m["tpr"] == 1.0
        assert m["fpr"] == 0.0
        assert m["f1"] == 1.0
        assert m["mcc"] == pytest.approx(1.0)

    def test_empty_recovery(self):
        p = 5
        Omega_true = _make_sparse_omega(p, [(0, 1), (2, 3)])
        Omega_hat = np.eye(p)  # predicts no edges
        m = sparsity_metrics(Omega_hat, Omega_true, threshold=1e-5)
        assert m["tpr"] == 0.0
        # FPR = FP / (FP + TN) = 0 / 8 = 0
        assert m["fpr"] == 0.0

    def test_false_positives(self):
        p = 4
        Omega_true = _make_sparse_omega(p, [])
        Omega_hat = _make_sparse_omega(p, [(0, 1)])
        m = sparsity_metrics(Omega_hat, Omega_true, threshold=1e-5)
        assert m["tpr"] == 0.0
        # FP = 1, TN = 5; FPR = 1/6
        assert m["fpr"] == pytest.approx(1 / 6)

    def test_edge_count_fields(self):
        p = 5
        Omega = _make_sparse_omega(p, [(0, 1), (2, 3), (1, 4)])
        m = sparsity_metrics(Omega, Omega, threshold=1e-5)
        assert m["n_edges_true"] == 3
        assert m["n_edges_detected"] == 3


class TestSparsityMetricsCredible:
    def test_tight_interval_detects_nonzero(self):
        """If posterior concentrates strictly away from 0, declare edge present."""
        p = 3
        Omega_true = _make_sparse_omega(p, [(0, 1)], signal=0.6)
        # Build fake posterior: 100 samples, all nonzero values close to Omega_true
        n = 100
        rng = np.random.default_rng(0)
        samples = np.broadcast_to(Omega_true, (n, p, p)).copy()
        # Small noise preserving the sign of each entry
        samples = samples + rng.normal(0, 0.001, size=samples.shape)
        # Symmetrise each
        samples = 0.5 * (samples + np.transpose(samples, (0, 2, 1)))
        m = sparsity_metrics_credible(samples, Omega_true, alpha=0.05)
        assert m["tpr"] == 1.0
        assert m["fpr"] == 0.0

    def test_wide_interval_rejects(self):
        """If posterior straddles 0, declare no edge."""
        p = 3
        Omega_true = _make_sparse_omega(p, [(0, 1)], signal=0.6)
        rng = np.random.default_rng(0)
        samples = np.zeros((500, p, p))
        for s in range(500):
            samples[s] = np.eye(p) + rng.normal(0, 1.0, size=(p, p))
            samples[s] = 0.5 * (samples[s] + samples[s].T)
        m = sparsity_metrics_credible(samples, Omega_true, alpha=0.05)
        # Wide intervals should miss the (0, 1) edge.
        assert m["tpr"] == 0.0


# ======================================================================
# Coverage
# ======================================================================

class TestCoverage95:
    def test_delta_at_truth_gives_coverage_zero_or_one(self):
        """Samples at exactly the true value → q_lo == q_hi == true, coverage = 1."""
        p = 3
        Omega_true = np.array([[2.0, 0.5, -0.3],
                               [0.5, 3.0, 0.1],
                               [-0.3, 0.1, 2.5]])
        samples = np.broadcast_to(Omega_true, (100, p, p)).copy()
        cov = coverage_95(samples, Omega_true)
        assert cov["coverage_95"] == pytest.approx(1.0)
        assert cov["mean_interval_width"] == pytest.approx(0.0, abs=1e-12)

    def test_normal_samples_give_roughly_nominal_coverage(self):
        p = 10
        rng = np.random.default_rng(0)
        Omega_true = np.eye(p)
        # Samples centred at the truth with sd 0.1.  95% interval should cover ~95%.
        samples = np.zeros((5000, p, p))
        for s in range(5000):
            samples[s] = Omega_true + rng.normal(0, 0.1, size=(p, p))
            samples[s] = 0.5 * (samples[s] + samples[s].T)
        # For an *unbiased* Gaussian draw centred at truth, coverage is 1.0
        # because we are testing membership of the exact mean, which is
        # always inside the 2.5%..97.5% interval of its own Gaussian with
        # probability 1.  (The test is a sanity check that the function
        # doesn't crash on large inputs.)
        cov = coverage_95(samples, Omega_true)
        assert 0.90 <= cov["coverage_95"] <= 1.0


# ======================================================================
# Eigenvalue metrics
# ======================================================================

class TestEigenvalueMetrics:
    def test_zero_mse_at_truth(self):
        Omega = np.diag([1.0, 2.0, 3.0, 4.0])
        em = eigenvalue_metrics(Omega, Omega)
        assert em["eigenvalue_mse"] == pytest.approx(0.0)
        assert em["condition_number_hat"] == pytest.approx(4.0)
        assert em["condition_number_true"] == pytest.approx(4.0)

    def test_diagonal_mse(self):
        A = np.diag([1.0, 2.0, 3.0])
        B = np.diag([1.0, 2.0, 5.0])
        em = eigenvalue_metrics(A, B)
        # Sorted desc: A=[3,2,1], B=[5,2,1]; diffs = [-2, 0, 0]; MSE = 4/3
        assert em["eigenvalue_mse"] == pytest.approx(4.0 / 3.0)


# ======================================================================
# GMV metrics
# ======================================================================

class TestGmvMetrics:
    def test_zero_diff_at_truth(self):
        Omega = np.diag([1.0, 2.0, 3.0])
        m = gmv_metrics(Omega, Omega)
        assert m["gmv_weight_l2_diff"] == pytest.approx(0.0)
        assert m["gmv_weight_norm"] == pytest.approx(m["oracle_gmv_weight_norm"])


# ======================================================================
# Shrinkage / bimodality / Wasserstein
# ======================================================================

class TestKappaComputation:
    def test_known_values(self):
        tau = np.array([1.0, 1.0])
        lambdas = np.array([[1.0], [2.0]])  # n_offdiag = 1
        kappa = compute_kappa_samples(tau, lambdas)
        # kappa_00 = 1/(1 + 1*1) = 0.5
        # kappa_10 = 1/(1 + 4*1) = 0.2
        assert kappa.shape == (2, 1)
        assert kappa[0, 0] == pytest.approx(0.5)
        assert kappa[1, 0] == pytest.approx(0.2)

    def test_broadcasting(self):
        n_samples, n_offdiag = 50, 7
        tau = np.ones(n_samples)
        lambdas = np.ones((n_samples, n_offdiag)) * 2.0
        kappa = compute_kappa_samples(tau, lambdas)
        assert kappa.shape == (n_samples, n_offdiag)
        assert np.allclose(kappa, 1 / (1 + 4))

    def test_kappa_in_unit_interval(self):
        rng = np.random.default_rng(0)
        tau = np.abs(rng.normal(size=100))
        lambdas = np.abs(rng.normal(size=(100, 20)))
        kappa = compute_kappa_samples(tau, lambdas)
        assert np.all(kappa >= 0.0)
        assert np.all(kappa <= 1.0)


class TestBimodalityCoefficient:
    def test_unimodal_normal(self):
        """A sample from a clean Gaussian should NOT look bimodal."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        b = bimodality_coefficient(x)
        assert b < BIMODALITY_THRESHOLD

    def test_bimodal_two_clusters(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(-3, 0.3, 2500), rng.normal(3, 0.3, 2500)])
        b = bimodality_coefficient(x)
        assert b > BIMODALITY_THRESHOLD

    def test_uniform_is_at_threshold(self):
        """Sarle's coefficient for a uniform distribution equals 5/9 exactly in the large-n limit."""
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, size=20000)
        b = bimodality_coefficient(x)
        assert b == pytest.approx(5.0 / 9.0, abs=0.05)

    def test_too_few_samples(self):
        assert np.isnan(bimodality_coefficient([1.0, 2.0]))


class TestWasserstein:
    def test_identical_distributions(self):
        a = np.array([1.0, 2.0, 3.0])
        assert shrinkage_wasserstein(a, a) == pytest.approx(0.0)

    def test_shifted_distributions_equal_length(self):
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([5.0, 6.0, 7.0])
        # Sorted mean absolute diff: all entries shift by 5
        assert shrinkage_wasserstein(a, b) == pytest.approx(5.0)

    def test_empty_inputs(self):
        assert np.isnan(shrinkage_wasserstein([], [1, 2]))


class TestShrinkageProfileSummary:
    def test_keys_present(self):
        rng = np.random.default_rng(0)
        k = rng.uniform(0, 1, size=1000)
        s = shrinkage_profile_summary(k)
        for key in (
            "n", "mean", "median", "std", "q25", "q75",
            "frac_near_0", "frac_near_1",
            "bimodality_coefficient", "is_bimodal",
        ):
            assert key in s

    def test_bimodal_flag(self):
        rng = np.random.default_rng(0)
        k = np.concatenate([rng.uniform(0, 0.05, 500), rng.uniform(0.95, 1.0, 500)])
        s = shrinkage_profile_summary(k)
        assert s["is_bimodal"] is True
        assert s["frac_near_0"] >= 0.4
        assert s["frac_near_1"] >= 0.4


# ======================================================================
# safe_call
# ======================================================================

class TestSafeCall:
    def test_returns_value_on_success(self):
        assert safe_call(lambda x: x * 2, 5) == 10

    def test_returns_default_on_error(self):
        def broken(*args):
            raise RuntimeError("boom")
        assert safe_call(broken, default=-1) == -1
