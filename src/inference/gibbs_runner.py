"""Li, Craig, and Bhadra (2019) Gibbs sampler for the graphical horseshoe.

Implements the column-wise block Gibbs sampler that exploits the
conditional structure of the precision matrix Ω.  Each full sweep
updates all p columns of Ω, all p(p-1)/2 local shrinkage parameters
λ²_ij, and the global shrinkage τ² (plus their data-augmentation
auxiliaries ν_ij and ξ).

Key properties vs NUTS:
- Model-specific (hand-derived conditionals), not black-box.
- No gradients needed (pure NumPy, not JAX).
- O(p³) per sweep from p matrix inversions.
- 100% acceptance rate (Gibbs), but component-wise updates may mix
  slowly in correlated posteriors.
- Single chain with Geweke + ESS for convergence, not R̂.

Expected wall time (MIT Engaging, 1 CPU):
    p=10  → < 1 min     (7000 sweeps)
    p=50  → 10–30 min   (7000 sweeps)
    p=100 → 1–3 hours   (8000 sweeps)
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np


# ======================================================================
# Index mapping: (i, j) upper-triangular pairs ↔ flat index
# ======================================================================

def _build_index_maps(p: int):
    """Precompute mappings between matrix positions and flat arrays.

    Returns
    -------
    pair_to_flat : dict
        {(i, j): flat_index} for all i < j.
    col_lambda_indices : list of np.ndarray
        ``col_lambda_indices[j]`` gives the flat lambda indices for the
        p-1 off-diagonal entries of column j, in the same order as
        ``[0, ..., j-1, j+1, ..., p-1]``.
    """
    idx_i, idx_j = np.triu_indices(p, k=1)
    pair_to_flat = {}
    for k in range(len(idx_i)):
        pair_to_flat[(int(idx_i[k]), int(idx_j[k]))] = k

    col_lambda_indices = []
    for j in range(p):
        indices = []
        for i in range(p):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            indices.append(pair_to_flat[key])
        col_lambda_indices.append(np.array(indices, dtype=int))

    return pair_to_flat, col_lambda_indices


# ======================================================================
# InvGamma sampling: X ~ InvGamma(a, b) ⟺ 1/X ~ Gamma(a, scale=1/b)
# ======================================================================

def _sample_invgamma(rng, shape, rate):
    """Sample from InvGamma(shape, rate).

    Parameterisation: if X ~ InvGamma(a, b), then E[X] = b/(a-1) for a > 1.
    """
    # Gamma(a, scale=1/b) then invert.  Guard against zero.
    g = rng.gamma(shape=shape, scale=1.0 / np.maximum(rate, 1e-30))
    return 1.0 / np.maximum(g, 1e-30)


# ======================================================================
# PD check via Schur complement (O(p²) instead of O(p³))
# ======================================================================

def _schur_pd_check(omega_jj, omega_minus_j, Omega_minus_inv):
    """Check if Ω stays PD after inserting omega_{-j,j} into column j.

    Schur complement: Ω ≻ 0  iff  ω_jj - ω_{-j,j}ᵀ Ω_{-j,-j}⁻¹ ω_{-j,j} > 0.
    """
    quad = omega_minus_j @ Omega_minus_inv @ omega_minus_j
    return omega_jj - quad > 0


# ======================================================================
# Single column update
# ======================================================================

def _sample_column(
    j: int,
    Omega: np.ndarray,
    lambda_sq: np.ndarray,
    tau_sq: float,
    S: np.ndarray,
    T: int,
    p: int,
    col_lambda_idx: np.ndarray,
    rng: np.random.Generator,
    max_rejection: int = 200,
) -> Tuple[np.ndarray, float, int]:
    """Update column j of Ω (off-diagonal + diagonal).

    Returns
    -------
    omega_col : np.ndarray, shape (p-1,)
        New off-diagonal entries for column j.
    omega_jj : float
        New diagonal entry for column j.
    n_rejections : int
        Number of truncated-normal rejection attempts.
    """
    # --- Partition Ω and S around column j ---
    others = np.concatenate([np.arange(j), np.arange(j + 1, p)])
    Omega_minus = Omega[np.ix_(others, others)]        # (p-1, p-1) = A
    s_minus_j = S[others, j]                            # (p-1,)
    s_jj = S[j, j]

    # --- Prior precision for the off-diagonal entries of column j ---
    lam_sq_col = lambda_sq[col_lambda_idx]              # (p-1,)
    d_j = 1.0 / (lam_sq_col * tau_sq + 1e-30)          # (p-1,)
    D_j = np.diag(d_j)

    # --- Ω_{-j,-j}⁻¹ enters both the posterior precision AND the Schur check ---
    Omega_minus_inv = np.linalg.inv(Omega_minus)

    # --- Conditional covariance and mean ---
    # The Wang (2012) / Li et al. (2019) reparameterisation of the likelihood
    # gives γ = ω_{-j,j} | rest ~ N(μ, Σ) with
    #     Σ = (s_jj · A⁻¹ + D_j)⁻¹,        μ = -Σ · s_{-j,j}
    # (Derivation: the γ-dependent terms in log L are
    #   -(s_jj/2) γᵀ A⁻¹ γ - s_{-j,j}ᵀ γ,  combined with the N(0, diag(λ²τ²)) prior.)
    #
    # An earlier implementation used `s_jj·A + D_j` with μ and Σ divided by s_jj;
    # that shrinks μ by a factor of s_jj² ≈ 10⁴, producing posterior draws that
    # are numerically indistinguishable from 0.  See gibbs_runner tests for
    # regression coverage.
    C_j = np.linalg.inv(s_jj * Omega_minus_inv + D_j)  # (p-1, p-1)  ← precision inverted
    mu_j = -C_j @ s_minus_j                             # (p-1,)       ← NO /s_jj
    Sigma_j = C_j                                       # (p-1, p-1)   ← NO /s_jj

    # Symmetrise for numerical stability
    Sigma_j = 0.5 * (Sigma_j + Sigma_j.T)

    # --- Sample off-diagonal via truncated normal (rejection) ---
    n_rej = 0
    omega_col = mu_j.copy()  # fallback
    for attempt in range(max_rejection):
        proposal = rng.multivariate_normal(mu_j, Sigma_j)
        # Current diagonal stays the same for the PD check
        current_jj = Omega[j, j]
        if _schur_pd_check(current_jj, proposal, Omega_minus_inv):
            omega_col = proposal
            break
        n_rej += 1
    else:
        # All attempts rejected — keep the mean (safe, PD-preserving in expectation)
        warnings.warn(
            f"Gibbs: truncated normal rejection failed after {max_rejection} "
            f"attempts for column {j}. Using conditional mean as fallback."
        )

    # --- Sample diagonal (shifted Gamma) ---
    quad = omega_col @ Omega_minus_inv @ omega_col
    g = rng.gamma(shape=T / 2.0 + 1.0, scale=2.0 / s_jj)
    omega_jj = g + quad

    return omega_col, float(omega_jj), n_rej


# ======================================================================
# Full Gibbs sweep
# ======================================================================

def _gibbs_sweep(
    Omega: np.ndarray,
    lambda_sq: np.ndarray,
    nu: np.ndarray,
    tau_sq: float,
    xi: float,
    S: np.ndarray,
    T: int,
    p: int,
    n_offdiag: int,
    col_lambda_indices: list,
    pair_to_flat: dict,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """One complete Gibbs sweep.

    Returns the updated state ``(Omega, lambda_sq, nu, tau_sq, xi)``
    plus the total number of rejection attempts across all columns.
    """
    total_rejections = 0

    # --- Block 1: Update Ω column by column ---
    for j in range(p):
        others = np.concatenate([np.arange(j), np.arange(j + 1, p)])
        omega_col, omega_jj, n_rej = _sample_column(
            j, Omega, lambda_sq, tau_sq, S, T, p,
            col_lambda_indices[j], rng,
        )
        total_rejections += n_rej

        # Insert into Omega
        Omega[others, j] = omega_col
        Omega[j, others] = omega_col
        Omega[j, j] = omega_jj

    # --- Block 2: Update local shrinkage λ²_ij via data augmentation ---
    idx_i, idx_j = np.triu_indices(p, k=1)
    omega_offdiag = Omega[idx_i, idx_j]  # flat vector of off-diagonal entries

    for k in range(n_offdiag):
        lsq = lambda_sq[k]
        nu[k] = _sample_invgamma(rng, shape=1.0, rate=1.0 + 1.0 / (lsq + 1e-30))
        lambda_sq[k] = _sample_invgamma(
            rng,
            shape=1.0,
            rate=1.0 / (nu[k] + 1e-30) + omega_offdiag[k] ** 2 / (2.0 * tau_sq + 1e-30),
        )

    # --- Block 3: Update global shrinkage τ² ---
    sum_ratio = np.sum(omega_offdiag ** 2 / (lambda_sq + 1e-30))
    xi = _sample_invgamma(rng, shape=1.0, rate=1.0 + 1.0 / (tau_sq + 1e-30))
    tau_sq = _sample_invgamma(
        rng,
        shape=(n_offdiag + 1.0) / 2.0,
        rate=1.0 / (xi + 1e-30) + 0.5 * sum_ratio,
    )

    return Omega, lambda_sq, nu, float(tau_sq), float(xi), total_rejections


# ======================================================================
# Convergence diagnostics (single chain)
# ======================================================================

def _ess_from_acf(x: np.ndarray) -> float:
    """Effective sample size from autocorrelation (initial positive sequence)."""
    n = len(x)
    if n < 4:
        return float(n)
    x_centered = x - x.mean()
    var = x_centered.var()
    if var == 0:
        return float(n)

    # FFT-based autocorrelation
    padded = np.zeros(2 * n)
    padded[:n] = x_centered
    f = np.fft.rfft(padded)
    acf_full = np.fft.irfft(f * np.conj(f))[:n] / (var * n)

    # Sum ACF using the initial positive sequence estimator (Geyer 1992):
    # pair consecutive lags and stop when a pair sum is negative.
    rho_sum = 0.0
    for k in range(1, n // 2):
        pair = acf_full[2 * k - 1] + acf_full[2 * k]
        if pair < 0:
            break
        rho_sum += pair

    ess = n / (1.0 + 2.0 * rho_sum)
    return max(1.0, float(ess))


def _geweke(x: np.ndarray, first_frac=0.1, last_frac=0.5):
    """Geweke convergence diagnostic: z-score and p-value."""
    n = len(x)
    n_first = max(int(n * first_frac), 2)
    n_last = max(int(n * last_frac), 2)
    x_first = x[:n_first]
    x_last = x[-n_last:]
    se = np.sqrt(x_first.var(ddof=1) / n_first + x_last.var(ddof=1) / n_last)
    if se == 0:
        return 0.0, 1.0
    z = (x_first.mean() - x_last.mean()) / se
    # Two-sided p-value from standard normal
    p_value = float(2.0 * (1.0 - _normal_cdf(abs(z))))
    return float(z), p_value


def _normal_cdf(x):
    """Standard normal CDF via the error function (no scipy needed)."""
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ======================================================================
# Public entry point
# ======================================================================

def _gibbs_defaults_for_p(p: int) -> dict:
    """Default Gibbs configuration scaled by dimension."""
    if p <= 10:
        return {"n_burnin": 1000, "n_samples": 5000, "n_thinning": 1}
    if p <= 50:
        return {"n_burnin": 2000, "n_samples": 5000, "n_thinning": 1}
    return {"n_burnin": 3000, "n_samples": 5000, "n_thinning": 2}


def run_gibbs(
    Y: np.ndarray,
    p: int,
    n_burnin: Optional[int] = None,
    n_samples: Optional[int] = None,
    n_thinning: Optional[int] = None,
    rng_seed: int = 0,
    max_rejection: int = 200,
) -> dict:
    """Run the Li et al. (2019) Gibbs sampler.

    Parameters
    ----------
    Y : np.ndarray, shape (T, p)
        Zero-mean observation matrix.
    p : int
        Dimension.
    n_burnin : int, optional
        Burn-in sweeps (discarded).  Auto-scaled by p if None.
    n_samples : int, optional
        Post-burn-in sweeps to collect.
    n_thinning : int, optional
        Keep every ``n_thinning``-th sample.
    rng_seed : int
    max_rejection : int
        Max truncated-normal rejection attempts per column update.

    Returns
    -------
    dict with keys:
        ``omega_samples, tau_sq_samples, lambda_sq_samples,
        omega_hat, diagnostics``.
    """
    T = Y.shape[0]
    S = Y.T @ Y  # scatter matrix (p, p)
    n_offdiag = p * (p - 1) // 2
    rng = np.random.default_rng(rng_seed)

    # --- Defaults ---
    defaults = _gibbs_defaults_for_p(p)
    if n_burnin is None:
        n_burnin = defaults["n_burnin"]
    if n_samples is None:
        n_samples = defaults["n_samples"]
    if n_thinning is None:
        n_thinning = defaults["n_thinning"]
    total_sweeps = n_burnin + n_samples * n_thinning
    n_saved = n_samples

    # --- Index maps ---
    pair_to_flat, col_lambda_indices = _build_index_maps(p)

    # --- Initialisation (PD + warm-started off-diagonals) ---
    #
    # A cold start with ``Omega = diag(...)`` makes sum_ratio = Σ ω²/λ²
    # start at exactly 0, which lets the ξ/τ² auxiliary feedback collapse
    # τ² multiplicatively to machine zero within a handful of sweeps
    # (each ξ update produces a large ξ because 1 + 1/τ² explodes; then
    # 1/ξ → 0 becomes the dominant term in the τ² rate; τ² shrinks
    # further; repeat).  Once τ² is ~1e-16, the prior precision
    # 1/(λ²·τ²) ≈ 1e15 dominates the likelihood and off-diagonals never
    # escape 0 — an absorbing state.
    #
    # Warm-start Ω from a ridge-regularised sample precision so
    # sum_ratio has real signal from sweep 1.
    try:
        S_ridge = S / float(T) + 0.1 * np.eye(p)
        Omega = np.linalg.inv(S_ridge)
        # Clamp diagonal to a safe range; off-diagonals preserved.
        np.fill_diagonal(Omega, np.clip(np.diag(Omega), 0.5, 10.0))
        # Guard against any residual non-PD after the diagonal clamp.
        min_eig = float(np.linalg.eigvalsh(Omega).min())
        if min_eig < 1e-3:
            Omega = Omega + (1e-3 - min_eig) * np.eye(p)
    except np.linalg.LinAlgError:
        diag_init = rng.gamma(shape=2.0, scale=0.5, size=p) + 1.0
        Omega = np.diag(diag_init)

    lambda_sq = np.ones(n_offdiag)
    nu = np.ones(n_offdiag)
    tau_sq = 1.0
    xi = 1.0

    # --- Storage ---
    omega_store = np.zeros((n_saved, p, p), dtype=np.float32)
    tau_sq_store = np.zeros(n_saved)
    lambda_sq_store = np.zeros((n_saved, n_offdiag))
    rejection_counts = []

    # --- Run ---
    start = time.time()
    sample_idx = 0
    for sweep in range(total_sweeps):
        Omega, lambda_sq, nu, tau_sq, xi, n_rej = _gibbs_sweep(
            Omega, lambda_sq, nu, tau_sq, xi,
            S, T, p, n_offdiag, col_lambda_indices, pair_to_flat, rng,
        )
        rejection_counts.append(n_rej)

        # Store after burn-in, respecting thinning
        if sweep >= n_burnin and (sweep - n_burnin) % n_thinning == 0:
            if sample_idx < n_saved:
                omega_store[sample_idx] = Omega.astype(np.float32)
                tau_sq_store[sample_idx] = tau_sq
                lambda_sq_store[sample_idx] = lambda_sq.copy()
                sample_idx += 1

        # Progress reporting
        if (sweep + 1) % max(1, total_sweeps // 10) == 0:
            elapsed = time.time() - start
            phase = "burn-in" if sweep < n_burnin else "sampling"
            print(
                f"  Gibbs sweep {sweep+1}/{total_sweeps} ({phase}), "
                f"tau²={tau_sq:.4f}, rejections={n_rej}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - start

    # --- Post-burn-in rejection statistics ---
    post_burnin_rej = rejection_counts[n_burnin:]
    per_column_rej_rate = np.array(post_burnin_rej) / p  # per sweep, per column

    # --- Convergence diagnostics ---
    # ESS on tau² trace (the slowest-mixing scalar)
    ess_tau = _ess_from_acf(tau_sq_store)

    # ESS on a few representative omega entries
    idx_i, idx_j = np.triu_indices(p, k=1)
    ess_omegas = []
    for k in range(min(20, n_offdiag)):
        ess_k = _ess_from_acf(omega_store[:, idx_i[k], idx_j[k]].astype(np.float64))
        ess_omegas.append(ess_k)
    min_ess_omega = min(ess_omegas) if ess_omegas else 0.0

    # Geweke on tau²
    geweke_z, geweke_p = _geweke(tau_sq_store)

    diagnostics = {
        "n_burnin": int(n_burnin),
        "n_samples": int(n_saved),
        "n_thinning": int(n_thinning),
        "n_total_sweeps": int(total_sweeps),
        "n_chains": 1,
        "n_rejection_failures": int(sum(1 for r in post_burnin_rej if r >= max_rejection * p)),
        "mean_rejection_rate_per_column": float(np.mean(per_column_rej_rate)),
        "max_rejection_rate_per_column": float(np.max(per_column_rej_rate)) if len(per_column_rej_rate) > 0 else 0.0,
        "min_ess_omega": float(min_ess_omega),
        "min_ess_tau": float(ess_tau),
        "geweke_z_tau": float(geweke_z),
        "geweke_p_value_tau": float(geweke_p),
        "elapsed_seconds": float(elapsed),
    }

    omega_hat = omega_store.mean(axis=0).astype(np.float64)

    return {
        "omega_samples": omega_store,
        "tau_sq_samples": tau_sq_store,
        "lambda_sq_samples": lambda_sq_store,
        "omega_hat": omega_hat,
        "diagnostics": diagnostics,
    }
