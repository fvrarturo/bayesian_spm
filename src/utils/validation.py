"""Validation functions for synthetic data generation.

Every generated ``(Omega, Sigma, Y)`` triple must pass these checks
before being saved to disk.  See ``_info/WORK1.md`` §5 for the full
spec.

Each validator returns a 2-tuple ``(ok: bool, diagnostics: dict)``.
When ``ok`` is False, the ``diagnostics`` dict contains an ``error``
key explaining why the check failed.  Warnings (non-fatal anomalies
like high condition numbers) are collected in ``diagnostics["warnings"]``
regardless of the ``ok`` status.
"""

from typing import Optional, Set, Tuple

import numpy as np


# Tolerances are centralized here so they can be audited in one place.
SYMMETRY_ATOL = 1e-12
PD_FLOOR = 0.1            # matches the floor enforced in _graph_to_omega
PD_FLOOR_TOL = 1e-10      # allow tiny floating-point dip below PD_FLOOR
NONZERO_TOL = 1e-12       # entries with |w| <= this are considered zero
RECONSTRUCTION_ATOL = 1e-8
COND_NUMBER_WARN = 1000.0
DIAG_SHIFT_WARN = 2.0
ZERO_MEAN_MAX_Z = 5.0     # 5-sigma per-component tolerance


# ----------------------------------------------------------------------
# Omega
# ----------------------------------------------------------------------

def validate_omega(
    Omega: np.ndarray,
    edge_set: Set[Tuple[int, int]],
    p: int,
    diagonal_shift: Optional[float] = None,
) -> Tuple[bool, dict]:
    """Check that Omega is symmetric, PD, and has the expected sparsity pattern.

    Parameters
    ----------
    Omega : np.ndarray, shape (p, p)
    edge_set : set of (i, j) tuples with i < j
        Ground truth: the positions of nonzero off-diagonal entries.
    p : int
    diagonal_shift : float, optional
        If provided, trigger a warning when the shift exceeds
        ``DIAG_SHIFT_WARN``.

    Returns
    -------
    (ok, diagnostics)
    """
    diagnostics: dict = {"warnings": []}

    # 1. Shape
    if Omega.shape != (p, p):
        return False, {
            **diagnostics,
            "error": f"Omega.shape {Omega.shape} != ({p}, {p})",
        }

    # 2. Finite
    if not np.isfinite(Omega).all():
        return False, {**diagnostics, "error": "Omega contains NaN or Inf"}

    # 3. Symmetric
    if not np.allclose(Omega, Omega.T, atol=SYMMETRY_ATOL):
        asym = float(np.max(np.abs(Omega - Omega.T)))
        return False, {
            **diagnostics,
            "error": f"Omega is not symmetric (max asymmetry = {asym:.2e})",
        }

    # 4. Positive definite, with a floor at PD_FLOOR.
    eigs = np.linalg.eigvalsh(Omega)
    eigmin = float(eigs.min())
    eigmax = float(eigs.max())
    diagnostics["min_eigenvalue"] = eigmin
    diagnostics["max_eigenvalue"] = eigmax

    if eigmin <= 0:
        return False, {**diagnostics, "error": f"Omega not PD (min eig = {eigmin:.2e})"}
    if eigmin < PD_FLOOR - PD_FLOOR_TOL:
        return False, {
            **diagnostics,
            "error": (
                f"Omega min eig {eigmin:.6f} < PD floor {PD_FLOOR} "
                f"(tol {PD_FLOOR_TOL})"
            ),
        }

    # 5. Sparsity pattern matches the claimed edge set.
    idx = np.triu_indices(p, k=1)
    actual_edges: Set[Tuple[int, int]] = set()
    for k in range(len(idx[0])):
        i = int(idx[0][k])
        j = int(idx[1][k])
        if abs(float(Omega[i, j])) > NONZERO_TOL:
            actual_edges.add((i, j))

    if actual_edges != edge_set:
        missing = edge_set - actual_edges
        extra = actual_edges - edge_set
        return False, {
            **diagnostics,
            "error": (
                f"sparsity pattern mismatch: "
                f"{len(missing)} edges missing, {len(extra)} unexpected"
            ),
            "missing_edges": sorted(missing)[:10],
            "extra_edges": sorted(extra)[:10],
        }

    # 6. Condition number warning
    cond = eigmax / eigmin
    diagnostics["condition_number"] = float(cond)
    if cond > COND_NUMBER_WARN:
        diagnostics["warnings"].append(
            f"high condition number: {cond:.1f} > {COND_NUMBER_WARN}"
        )

    # 7. Diagonal shift warning
    if diagonal_shift is not None and diagonal_shift > DIAG_SHIFT_WARN:
        diagnostics["warnings"].append(
            f"large diagonal shift: {diagonal_shift:.2f} > {DIAG_SHIFT_WARN}"
        )

    return True, diagnostics


# ----------------------------------------------------------------------
# Sigma = Omega^{-1}
# ----------------------------------------------------------------------

def validate_sigma(Sigma: np.ndarray, Omega: np.ndarray, p: int) -> Tuple[bool, dict]:
    """Check that Sigma = Omega^{-1} to within tolerance and is PD."""
    diagnostics: dict = {"warnings": []}

    if Sigma.shape != (p, p):
        return False, {
            **diagnostics,
            "error": f"Sigma.shape {Sigma.shape} != ({p}, {p})",
        }
    if not np.isfinite(Sigma).all():
        return False, {**diagnostics, "error": "Sigma contains NaN or Inf"}

    recon = Omega @ Sigma
    recon_err = float(np.max(np.abs(recon - np.eye(p))))
    diagnostics["reconstruction_max_err"] = recon_err
    if recon_err > RECONSTRUCTION_ATOL:
        return False, {
            **diagnostics,
            "error": (
                f"Omega @ Sigma != I (max err = {recon_err:.2e} > "
                f"{RECONSTRUCTION_ATOL})"
            ),
        }

    eigs = np.linalg.eigvalsh(Sigma)
    eigmin = float(eigs.min())
    diagnostics["sigma_min_eigenvalue"] = eigmin
    if eigmin <= 0:
        return False, {
            **diagnostics,
            "error": f"Sigma not PD (min eig = {eigmin:.2e})",
        }

    return True, diagnostics


# ----------------------------------------------------------------------
# Y
# ----------------------------------------------------------------------

def validate_data(
    Y: np.ndarray,
    T: int,
    p: int,
    Sigma: Optional[np.ndarray] = None,
) -> Tuple[bool, dict]:
    """Check the sampled data matrix Y.

    Hard failures:
    - wrong shape
    - contains NaN or Inf
    - sample covariance rank != min(T-1, p)

    Warnings (do not fail):
    - max per-component z-score on the sample mean exceeds 5, computed as
      ``|ybar_i| / sqrt(Sigma_{ii} / T)``.  Only checked when Sigma is
      provided, because absent Sigma we cannot scale properly.
    """
    diagnostics: dict = {"warnings": []}

    if Y.shape != (T, p):
        return False, {
            **diagnostics,
            "error": f"Y.shape {Y.shape} != ({T}, {p})",
        }
    if not np.isfinite(Y).all():
        return False, {**diagnostics, "error": "Y contains NaN or Inf"}

    # Sample covariance and rank
    Sigma_hat = np.cov(Y, rowvar=False, bias=False)
    actual_rank = int(np.linalg.matrix_rank(Sigma_hat))
    expected_rank = int(min(T - 1, p))
    diagnostics["sample_cov_rank"] = actual_rank
    diagnostics["expected_rank"] = expected_rank

    if actual_rank != expected_rank:
        return False, {
            **diagnostics,
            "error": (
                f"sample cov rank {actual_rank} != expected {expected_rank}"
            ),
        }

    # Zero-mean z-score check (warning only)
    ybar = Y.mean(axis=0)
    diagnostics["max_abs_mean"] = float(np.max(np.abs(ybar)))

    if Sigma is not None:
        per_component_std = np.sqrt(np.diag(Sigma) / T)
        # Guard against a zero diagonal (impossible for PD Sigma but defensive)
        per_component_std = np.where(per_component_std > 0, per_component_std, 1.0)
        z_scores = np.abs(ybar) / per_component_std
        max_z = float(np.max(z_scores))
        diagnostics["max_mean_zscore"] = max_z
        if max_z > ZERO_MEAN_MAX_Z:
            diagnostics["warnings"].append(
                f"max mean z-score {max_z:.2f} > {ZERO_MEAN_MAX_Z}"
            )

    return True, diagnostics
