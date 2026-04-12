"""Evaluation dispatcher.

One public function, ``evaluate``, that:

1. Loads the ground-truth Ω₀ from a WORK1 seed directory.
2. Loads the method estimate Ω̂ (and posterior samples, if applicable)
   from a WORK2 method directory.
3. Computes every metric in the WORK2 §3.3 schema.
4. Writes ``metrics.json`` atomically next to the diagnostics.
5. Returns the metrics dict.

For **failed / timeout / singular** runs, the metrics are all set to
``null`` but the file is still written, so downstream aggregators can
distinguish "not run yet" from "ran but failed".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.evaluation.metrics import (
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
from src.evaluation.shrinkage import (
    bimodality_coefficient,
    compute_kappa_hat,
    shrinkage_profile_summary,
    shrinkage_wasserstein,
)

# The canonical list of numeric metric keys that every metrics.json carries.
# Used to build null-filled metrics dicts for failed runs.
_NUMERIC_METRIC_KEYS = [
    "steins_loss",
    "frobenius_loss",
    "frobenius_loss_relative",
    "spectral_loss",
    "trace_error",
    "eigenvalue_mse",
    "condition_number_hat",
    "condition_number_true",
    "gmv_weight_norm",
    "oracle_gmv_weight_norm",
    "gmv_weight_l2_diff",
    "gmv_weight_l1_diff",
    "tpr",
    "fpr",
    "precision",
    "recall",
    "f1",
    "mcc",
    "n_edges_detected",
    "n_edges_true",
    "coverage_95",
    "mean_interval_width",
    "mean_posterior_std_offdiag",
    "bimodality_coefficient_kappa",
    "shrinkage_wasserstein_vs_nuts",
]


def _null_metrics(method: str, diagnostics: dict) -> dict:
    """Build a metrics dict with all numeric fields set to None."""
    out = {
        "method": method,
        "config_id": diagnostics.get("config_id"),
        "seed": diagnostics.get("seed"),
        "p": diagnostics.get("p"),
        "T": diagnostics.get("T"),
        "graph": diagnostics.get("graph"),
        "sparsity": diagnostics.get("sparsity"),
        "gamma": diagnostics.get("gamma"),
        "status": diagnostics.get("status", "unknown"),
        "edge_detection_method": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for k in _NUMERIC_METRIC_KEYS:
        out[k] = None
    return out


def _frequentist_threshold_for_method(method: str) -> float:
    """Detection threshold for each frequentist method."""
    if method == "glasso":
        return 0.0  # exact structural zeros
    return 1e-5


def _compute_bayesian_extras(
    metrics: dict,
    omega_samples: np.ndarray,
    Omega_true: np.ndarray,
    kappa_samples: Optional[np.ndarray],
    method: str,
    results_dir: Path,
) -> None:
    """Fill in Bayesian-only metrics: credible-interval sparsity, coverage,
    bimodality, Wasserstein-vs-NUTS."""
    # --- Credible-interval sparsity (primary for Bayesian methods) ---
    ci_metrics = sparsity_metrics_credible(omega_samples, Omega_true, alpha=0.05)
    metrics.update(ci_metrics)
    metrics["edge_detection_method"] = "credible_interval_95"

    # --- Coverage ---
    cov = coverage_95(omega_samples, Omega_true)
    metrics.update(cov)

    # --- Kappa / bimodality ---
    if kappa_samples is not None:
        kappa_hat = compute_kappa_hat(kappa_samples)
        metrics["bimodality_coefficient_kappa"] = bimodality_coefficient(kappa_hat)
        # Extended summary as a nested dict for downstream diagnostics.
        metrics["shrinkage_profile"] = shrinkage_profile_summary(kappa_hat)
    else:
        metrics["bimodality_coefficient_kappa"] = None

    # --- Wasserstein-vs-NUTS ---
    if method == "nuts":
        metrics["shrinkage_wasserstein_vs_nuts"] = None
    else:
        nuts_kappa_path = results_dir.parent / "nuts" / "kappa_samples.npy"
        if kappa_samples is not None and nuts_kappa_path.exists():
            try:
                nuts_kappa_samples = np.load(nuts_kappa_path)
                nuts_kappa_hat = compute_kappa_hat(nuts_kappa_samples)
                this_kappa_hat = compute_kappa_hat(kappa_samples)
                metrics["shrinkage_wasserstein_vs_nuts"] = shrinkage_wasserstein(
                    nuts_kappa_hat, this_kappa_hat
                )
            except Exception as e:
                metrics["shrinkage_wasserstein_vs_nuts"] = None
                metrics["shrinkage_wasserstein_error"] = repr(e)
        else:
            metrics["shrinkage_wasserstein_vs_nuts"] = None


def evaluate(
    method: str,
    data_dir: Path,
    results_dir: Path,
    true_threshold: float = 1e-5,
) -> dict:
    """Compute the full metric suite for one (method, seed) pair.

    Parameters
    ----------
    method : str
    data_dir : Path
        WORK1 seed directory (contains ``omega_true.npy``, ``metadata.json``).
    results_dir : Path
        WORK2 method directory (contains ``diagnostics.json`` and
        ``omega_hat.npy``, optionally ``omega_samples.npy``,
        ``kappa_samples.npy``).
    true_threshold : float
        Threshold below which ground-truth entries are treated as zero
        when computing sparsity metrics.

    Returns
    -------
    metrics : dict
        Also written to ``<results_dir>/metrics.json``.
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)

    # --- 1. Load diagnostics to determine status ---
    diag_path = results_dir / "diagnostics.json"
    if not diag_path.exists():
        raise FileNotFoundError(f"diagnostics.json not found in {results_dir}")
    with open(diag_path) as f:
        diagnostics = json.load(f)

    status = diagnostics.get("status", "unknown")

    # --- 2. Non-success: write null metrics and exit ---
    if status != "success":
        metrics = _null_metrics(method, diagnostics)
        _write_metrics(results_dir, metrics)
        return metrics

    # --- 3. Load ground truth ---
    Omega_true = np.load(data_dir / "omega_true.npy")
    with open(data_dir / "metadata.json") as f:
        data_metadata = json.load(f)

    # --- 4. Load inference estimate ---
    omega_hat_path = results_dir / "omega_hat.npy"
    if not omega_hat_path.exists():
        metrics = _null_metrics(method, diagnostics)
        metrics["status"] = "success_but_no_estimate"
        _write_metrics(results_dir, metrics)
        return metrics

    Omega_hat = np.load(omega_hat_path)

    # --- 5. Point-estimate metrics (common to all methods) ---
    metrics: dict = {
        "method": method,
        "config_id": int(diagnostics["config_id"]),
        "seed": int(diagnostics["seed"]),
        "p": int(diagnostics.get("p", Omega_hat.shape[0])),
        "T": int(diagnostics.get("T", 0)),
        "graph": data_metadata.get("graph"),
        "sparsity": data_metadata.get("sparsity"),
        "gamma": data_metadata.get("gamma"),
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steins_loss": safe_call(steins_loss, Omega_hat, Omega_true),
        "frobenius_loss": frobenius_loss(Omega_hat, Omega_true),
        "frobenius_loss_relative": frobenius_loss_relative(Omega_hat, Omega_true),
        "spectral_loss": spectral_loss(Omega_hat, Omega_true),
        "trace_error": trace_error(Omega_hat, Omega_true),
    }
    metrics.update(eigenvalue_metrics(Omega_hat, Omega_true))
    metrics.update(gmv_metrics(Omega_hat, Omega_true))

    # --- 6. Threshold-based sparsity (always computed) ---
    threshold = _frequentist_threshold_for_method(method)
    sp_thresh = sparsity_metrics(Omega_hat, Omega_true, threshold=threshold)
    # Save the threshold variant under `_threshold` suffixed keys so we
    # don't overwrite when credible-interval metrics come in later.
    for k, v in sp_thresh.items():
        metrics[f"{k}_threshold"] = v
    metrics["threshold_used"] = threshold

    # --- 7. Bayesian-only extras ---
    omega_samples_path = results_dir / "omega_samples.npy"
    kappa_samples_path = results_dir / "kappa_samples.npy"
    has_samples = omega_samples_path.exists()

    if has_samples:
        omega_samples = np.load(omega_samples_path).astype(np.float64)
        kappa_samples = (
            np.load(kappa_samples_path).astype(np.float64)
            if kappa_samples_path.exists()
            else None
        )
        _compute_bayesian_extras(
            metrics, omega_samples, Omega_true, kappa_samples, method, results_dir
        )
    else:
        # Frequentist: threshold-based sparsity is primary.
        metrics.update(sp_thresh)
        metrics["edge_detection_method"] = f"threshold_{threshold}"
        # Fill Bayesian-only keys with None for schema consistency.
        metrics["coverage_95"] = None
        metrics["mean_interval_width"] = None
        metrics["mean_posterior_std_offdiag"] = None
        metrics["bimodality_coefficient_kappa"] = None
        metrics["shrinkage_wasserstein_vs_nuts"] = None

    # --- 8. Save and return ---
    _write_metrics(results_dir, metrics)
    return metrics


def _write_metrics(results_dir: Path, metrics: dict) -> None:
    """Write metrics.json atomically (tiny file; no .tmp juggling)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "metrics.json"
    # Convert numpy types to plain Python for JSON.
    from src.inference.run_single import _to_py
    with open(path, "w") as f:
        json.dump(_to_py(metrics), f, indent=2)
