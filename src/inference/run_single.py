"""Single-task inference dispatcher.

One public function, ``run_inference``, handles all six methods:

    nuts, advi_mf, advi_fr, glasso, ledoit_wolf, sample_cov

The dispatcher owns:

- Loading ``Y``, ``metadata`` from a WORK1 seed directory.
- Sizing method hyperparameters by dimension (e.g. reduce NUTS samples
  at p=10).
- Exception handling — any method failure produces a
  ``status="failed"`` diagnostics dict rather than a crash.
- Optional Linux ``SIGALRM`` timeout wrapping.
- Post-processing posterior samples (kappa, assembled Omega matrices,
  thinning, casting).
- Atomic disk writes (``.tmp`` rename pattern, same as WORK1).

Higher-level concerns (CLI, SLURM arrays, evaluation) live outside.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


# ======================================================================
# Public configuration
# ======================================================================

DEFAULT_NUTS_TIMEOUT_SECONDS = 42_000  # 11h40m (leaves 20 min buffer under 12h wall)
DEFAULT_ADVI_TIMEOUT_SECONDS = 20_000  # ~5.5h (leaves buffer under 6h wall)
DEFAULT_FREQ_TIMEOUT_SECONDS = 1_800   # 30 minutes
DEFAULT_MAX_SAVED_SAMPLES = 5_000      # cap on posterior-sample array length

# Methods that produce posterior samples (i.e. Bayesian).
BAYESIAN_METHODS = ("nuts", "gibbs", "advi_mf", "advi_fr", "advi_lr")

# Full-rank ADVI fallback threshold.  Above this latent dimension, we
# auto-fall-back to low-rank to keep memory manageable.
FULL_RANK_D_MAX = 5_000


# ======================================================================
# JSON-serialisation helpers
# ======================================================================

def _to_py(x):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, np.ndarray):
        return [_to_py(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


# ======================================================================
# Timeout (Linux SIGALRM)
# ======================================================================

class _Timeout(Exception):
    """Raised when a timed-out block runs longer than the alarm."""


def _run_with_timeout(fn, seconds, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` with a Linux SIGALRM timeout.

    On non-Linux platforms (e.g. macOS tests), runs ``fn`` without a
    timeout guard.  JAX's internal C code may or may not respond
    promptly to the alarm — the caller should treat this as a
    best-effort safeguard, not a hard guarantee.
    """
    if seconds is None or seconds <= 0 or platform.system() != "Linux":
        return fn(*args, **kwargs)

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ======================================================================
# Thinning
# ======================================================================

def _thin(arr, max_samples):
    """Uniformly thin along axis 0 to at most ``max_samples`` rows."""
    a = np.asarray(arr)
    n = a.shape[0]
    if n <= max_samples:
        return a
    # Uniform stride; always include the last sample for a tight bound.
    idx = np.round(np.linspace(0, n - 1, max_samples)).astype(int)
    return a[idx]


# ======================================================================
# Atomic save
# ======================================================================

_ATOMIC_SUFFIX = ".tmp"


def _save_results_atomic(
    output_dir: Path,
    result: dict,
    diagnostics: dict,
    max_saved_samples: int,
) -> None:
    """Write result arrays + diagnostics.json atomically.

    Files are written to ``output_dir.tmp/`` first, then ``.tmp`` is
    renamed on top of ``output_dir``.  If any step fails the .tmp
    directory is removed and the target is untouched.
    """
    output_dir = Path(output_dir)
    tmp_dir = output_dir.with_name(output_dir.name + _ATOMIC_SUFFIX)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Diagnostics always written, even for failed runs.
        with open(tmp_dir / "diagnostics.json", "w") as f:
            json.dump(_to_py(diagnostics), f, indent=2)

        # 2. Omega_hat (if present — skipped for failed runs with no estimate).
        omega_hat = result.get("omega_hat")
        if omega_hat is not None:
            np.save(
                tmp_dir / "omega_hat.npy",
                np.asarray(omega_hat, dtype=np.float64),
            )

        # 3. Bayesian: posterior samples.  Thin to cap disk usage.
        #    Raw parameter samples stay float64; the bulky omega_samples
        #    is stored in float32 to halve the footprint.
        sample_dtypes = {
            "omega_samples": np.float32,
            "tau_samples": np.float64,
            "lambda_samples": np.float64,
            "omega_diag_samples": np.float64,
            "kappa_samples": np.float64,
        }
        for key, dtype in sample_dtypes.items():
            arr = result.get(key)
            if arr is None:
                continue
            thinned = _thin(arr, max_saved_samples).astype(dtype, copy=False)
            np.save(tmp_dir / f"{key}.npy", thinned)

        # 4. ADVI: ELBO trace (negative-ELBO losses).
        elbo = result.get("elbo_trace")
        if elbo is not None:
            np.save(
                tmp_dir / "elbo_trace.npy",
                np.asarray(elbo, dtype=np.float32),
            )

        # 5. Sample-cov / frequentist extras.
        for key in ("sigma_hat",):
            arr = result.get(key)
            if arr is None:
                continue
            np.save(
                tmp_dir / f"{key}.npy",
                np.asarray(arr, dtype=np.float64),
            )

        # 6. Frequentist benchmark: offdiag magnitude vector for shrinkage-profile plots.
        mags = result.get("offdiag_magnitudes")
        if mags is not None:
            np.save(
                tmp_dir / "offdiag_magnitudes.npy",
                np.asarray(mags, dtype=np.float64),
            )

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Rename into place.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)


# ======================================================================
# Model wrapper (lets us swap ncp without touching the base model)
# ======================================================================

def _horseshoe_init_values(p, ncp=True):
    """Hand-picked PD-safe starting values for the graphical horseshoe.

    At p=100, ``init_to_median(num_samples=15)`` fails to find a valid
    starting point about 30% of the time because (a) each latent site's
    median is estimated from only 15 samples, (b) the HalfCauchy's heavy
    tail gives a noisy median, and (c) with 10,000+ latent dimensions,
    even one outlier is enough to push Omega out of the PD cone.

    Passing explicit values bypasses the sampling entirely:
        z=0, λ=1, τ=1, ω_ii=5
        ⟹ ω_{ij} = z·λ·τ = 0 for all i<j
        ⟹ Ω = 5·I (trivially PD, finite log-density)
    """
    import numpy as np
    n_offdiag = p * (p - 1) // 2
    values = {
        "tau": np.array(1.0),
        "lambdas": np.ones(n_offdiag),
        "omega_diag": np.ones(p) * 5.0,
    }
    if ncp:
        values["z"] = np.zeros(n_offdiag)
    else:
        values["omega_offdiag"] = np.zeros(n_offdiag)
    return values


def _make_horseshoe_model(ncp=True, tau_scale=1.0, diag_prior="halfnormal"):
    """Return a closure over the graphical-horseshoe model.

    NumPyro traces the model by calling it with positional and keyword
    arguments at ``mcmc.run(...)`` / ``svi.init(...)`` time.  By
    wrapping in a closure we can override the static options (``ncp``,
    ``tau_scale``, ``diag_prior``) without touching the base model
    definition or plumbing extra kwargs through every runner.
    """
    from src.models.graphical_horseshoe import graphical_horseshoe

    def _model(Y, p):
        return graphical_horseshoe(
            Y=Y, p=p, ncp=ncp, tau_scale=tau_scale, diag_prior=diag_prior
        )

    _model.__name__ = f"graphical_horseshoe_ncp{int(ncp)}"
    return _model


# ======================================================================
# Omega-sample reconstruction
# ======================================================================

def _reconstruct_samples(samples, p):
    """Given a NumPyro samples dict, return numpy arrays for
    (tau, lambdas, omega_diag, omega_offdiag).

    Handles both centered and non-centered parameterizations.
    """
    tau = np.asarray(samples["tau"])
    lambdas = np.asarray(samples["lambdas"])
    omega_diag = np.asarray(samples["omega_diag"])

    if "omega_offdiag" in samples:
        omega_offdiag = np.asarray(samples["omega_offdiag"])
    elif "z" in samples:
        z = np.asarray(samples["z"])
        omega_offdiag = z * lambdas * tau[:, None]
    else:
        raise KeyError(
            "Posterior samples missing both 'omega_offdiag' and 'z'; "
            "cannot reconstruct off-diagonal entries."
        )
    return tau, lambdas, omega_diag, omega_offdiag


def _assemble_omega_matrices(omega_offdiag, omega_diag, p):
    """Assemble full (n_samples, p, p) precision matrices from offdiag + diag."""
    n = omega_offdiag.shape[0]
    idx = np.triu_indices(p, k=1)
    out = np.zeros((n, p, p), dtype=np.float64)
    out[:, idx[0], idx[1]] = omega_offdiag
    out = out + np.transpose(out, (0, 2, 1))
    for s in range(n):
        np.fill_diagonal(out[s], omega_diag[s])
    return out


# ======================================================================
# Method: NUTS
# ======================================================================

def _nuts_defaults_for_p(p):
    """Default NUTS hyperparameters, tuned per-dimension.

    Small p uses a shorter budget AND a tighter target acceptance
    (0.95 instead of 0.85).  The horseshoe's heavy-tailed priors make
    the potential surface very steep near the PD boundary, so a small
    step size is worth the extra compute to avoid divergences.
    """
    if p <= 10:
        return {
            "num_warmup": 1000,
            "num_samples": 2000,
            "num_chains": 4,
            "target_accept_prob": 0.95,
            "max_tree_depth": 10,
        }
    return {
        "num_warmup": 2000,
        "num_samples": 5000,
        "num_chains": 4,
        "target_accept_prob": 0.90,
        "max_tree_depth": 10,
    }


def _run_nuts_core(Y, p, T, rng_seed, nuts_params, ncp):
    """Run NUTS with the given hyperparameters and parameterization."""
    import numpyro
    from numpyro.diagnostics import summary as nuts_summary
    from src.inference.nuts_runner import init_to_value, run_nuts

    # Defensive: disable global argument validation so NumPyro never
    # tries to check precision_matrix positive-definiteness at Python
    # level.  The model itself also passes validate_args=False on the
    # likelihood; this is a belt-and-braces second line of defence.
    try:
        numpyro.enable_validation(False)
    except Exception:
        pass

    # Use explicit PD-safe init values.  At p=100 the default
    # init_to_median(num_samples=15) fails ~30% of the time because the
    # 15-sample median of HalfCauchy draws is noisy enough to land
    # outside the PD cone when all p(p-1)/2 latents are combined into
    # Omega.  init_to_value(z=0, lambda=1, tau=1, omega_diag=5)
    # guarantees Omega = 5*I on the first step.
    init_strategy = init_to_value(values=_horseshoe_init_values(p, ncp=ncp))

    model = _make_horseshoe_model(ncp=ncp)
    mcmc = run_nuts(
        model=model,
        Y=Y,
        p=p,
        rng_seed=rng_seed,
        progress_bar=False,
        extra_fields=("diverging",),
        init_strategy=init_strategy,
        **nuts_params,
    )

    # --- Posterior samples ---
    samples = mcmc.get_samples()
    tau, lambdas, omega_diag_samples, omega_offdiag = _reconstruct_samples(samples, p)
    omega_samples = _assemble_omega_matrices(omega_offdiag, omega_diag_samples, p)

    # --- Convergence diagnostics ---
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    # Only tau and lambdas and omega_diag and z are meaningful for R-hat.
    # omega_offdiag is a deterministic site; summary may report NaNs for it.
    diag_sites = {
        k: v
        for k, v in samples_by_chain.items()
        if k in {"tau", "lambdas", "z", "omega_diag", "omega_offdiag"}
    }
    stats = nuts_summary(diag_sites)
    all_rhats = []
    all_ess = []
    for _, s in stats.items():
        if "r_hat" in s:
            arr = np.asarray(s["r_hat"]).ravel()
            all_rhats.extend([float(v) for v in arr if np.isfinite(v)])
        if "n_eff" in s:
            arr = np.asarray(s["n_eff"]).ravel()
            all_ess.extend([float(v) for v in arr if np.isfinite(v)])
    max_rhat = max(all_rhats) if all_rhats else float("nan")
    min_ess = min(all_ess) if all_ess else float("nan")

    # --- Divergences ---
    n_divergences = 0
    try:
        extra = mcmc.get_extra_fields()
        if "diverging" in extra:
            n_divergences = int(np.sum(np.asarray(extra["diverging"])))
    except Exception:
        n_divergences = 0

    # --- Kappa ---
    from src.evaluation.shrinkage import compute_kappa_samples
    kappa_samples = compute_kappa_samples(tau, lambdas)

    omega_hat = omega_samples.mean(axis=0)

    # For the graphical horseshoe, zero divergences is an unrealistic
    # bar — the local/global scale funnel produces a small number even
    # in well-mixing chains.  We accept anything under 5% of total
    # samples, which is the standard Stan threshold for hierarchical
    # shrinkage models.  R-hat and ESS are the primary convergence
    # gates; divergences above 10% trigger the NCP↔CP retry upstream.
    n_total = int(tau.shape[0])
    divergence_rate = float(n_divergences / max(n_total, 1))
    converged = (
        divergence_rate < 0.05
        and np.isfinite(max_rhat)
        and max_rhat < 1.01
        and np.isfinite(min_ess)
        and min_ess > 400
    )

    diagnostics = {
        "num_warmup": int(nuts_params["num_warmup"]),
        "num_samples": int(nuts_params["num_samples"]),
        "num_chains": int(nuts_params["num_chains"]),
        "target_accept_prob": float(nuts_params["target_accept_prob"]),
        "max_tree_depth": int(nuts_params["max_tree_depth"]),
        "parameterization": "ncp" if ncp else "cp",
        "n_divergences": int(n_divergences),
        "divergence_rate": divergence_rate,
        "max_rhat": float(max_rhat) if np.isfinite(max_rhat) else None,
        "min_bulk_ess": float(min_ess) if np.isfinite(min_ess) else None,
        "n_total_posterior_samples": n_total,
        "converged": bool(converged),
    }
    return {
        "status": "success",
        "omega_hat": omega_hat,
        "omega_samples": omega_samples,
        "tau_samples": tau,
        "lambda_samples": lambdas,
        "omega_diag_samples": omega_diag_samples,
        "kappa_samples": kappa_samples,
        "diagnostics": diagnostics,
    }


def _run_nuts(Y, p, T, timeout_seconds=None, rng_seed=0, **kwargs):
    """Top-level NUTS wrapper with divergence retry."""
    nuts_params = _nuts_defaults_for_p(p)
    for k in list(nuts_params.keys()):
        if k in kwargs:
            nuts_params[k] = kwargs[k]
    ncp = bool(kwargs.get("ncp", True))

    # First attempt
    result = _run_with_timeout(
        _run_nuts_core, timeout_seconds, Y, p, T, rng_seed, nuts_params, ncp,
    )

    # If more than 10% of samples diverged, retry with the opposite parameterization.
    diag = result.get("diagnostics", {})
    n_div = int(diag.get("n_divergences", 0) or 0)
    n_samp = int(nuts_params["num_samples"]) * int(nuts_params["num_chains"])
    if n_div > 0.1 * n_samp:
        diag["divergence_retry"] = True
        diag["divergence_retry_from_ncp"] = ncp
        retry_ncp = not ncp
        try:
            retry_result = _run_with_timeout(
                _run_nuts_core,
                timeout_seconds,
                Y, p, T, rng_seed + 1, nuts_params, retry_ncp,
            )
            retry_diag = retry_result.get("diagnostics", {})
            retry_n_div = int(retry_diag.get("n_divergences", 0) or 0)
            if retry_n_div < n_div:
                retry_diag["divergence_retry"] = True
                retry_diag["divergence_retry_from_ncp"] = ncp
                return retry_result
        except Exception as e:
            diag["divergence_retry_error"] = repr(e)

    return result


# ======================================================================
# Method: ADVI
# ======================================================================

def _run_advi_core(Y, p, T, guide_type, advi_kwargs, rng_seed):
    """Call the existing ADVI runner and extract samples."""
    import numpyro
    from src.inference.advi_runner import run_advi

    try:
        numpyro.enable_validation(False)
    except Exception:
        pass

    # Separate guide-specific kwargs from run_advi's own.
    low_rank = advi_kwargs.pop("low_rank", None)

    # PD-safe explicit init values (avoids the p=100
    # "Cannot find valid initial parameters" failure).
    init_values = _horseshoe_init_values(p, ncp=True)

    model = _make_horseshoe_model(ncp=True)
    advi_result = run_advi(
        model=model,
        Y=Y,
        p=p,
        guide_type=guide_type,
        rng_seed=rng_seed,
        low_rank=low_rank,
        init_values=init_values,
        **advi_kwargs,
    )
    return advi_result


def _run_advi(Y, p, T, guide_type, timeout_seconds=None, rng_seed=0, **kwargs):
    """Top-level ADVI wrapper with lr-retry and fr→lr fallback."""
    # --- Defaults scaled by p ---
    # At p=10 the model has ~100 latent parameters; 50k steps is massive
    # overkill (the ELBO plateaus by ~5k).  At p=50 or p=100 we need the
    # full budget to converge.
    if p <= 10:
        advi_defaults = {
            "num_steps": 10_000,
            "learning_rate": 0.005,
            "num_samples": 5000,
            "num_seeds": 3,
        }
    else:
        advi_defaults = {
            "num_steps": 50_000,
            "learning_rate": 0.005,
            "num_samples": 5000,
            "num_seeds": 5,
        }

    if guide_type == "full_rank":
        # Full-rank is more expensive per step and needs more steps to
        # converge, but we still want p-sized ceilings.
        advi_defaults["num_steps"] = max(20_000, 2 * advi_defaults["num_steps"])
        advi_defaults["learning_rate"] = 0.005
        advi_defaults["num_seeds"] = max(2, advi_defaults["num_seeds"] - 2)
    elif guide_type == "low_rank":
        advi_defaults["num_steps"] = max(20_000, 2 * advi_defaults["num_steps"])
        advi_defaults["learning_rate"] = 0.005
        advi_defaults["num_seeds"] = max(2, advi_defaults["num_seeds"] - 2)

    advi_defaults.update({k: v for k, v in kwargs.items() if k in advi_defaults})

    # --- Feasibility check: fall back to low_rank for big full_rank ---
    D = p * p + 1  # approx latent count under ncp
    used_guide = guide_type
    fallback_to_lr = False
    low_rank = None
    if guide_type == "full_rank" and D > FULL_RANK_D_MAX:
        used_guide = "low_rank"
        fallback_to_lr = True
        # Low-rank tends to want more restarts, not fewer.
        advi_defaults.setdefault("num_seeds", 3)
        # Pick a rank that scales with p but caps far below the full D.
        low_rank = min(max(int(p / 2), 25), 100)

    # --- Retry schedule on nan/inf loss ---
    initial_lr = advi_defaults["learning_rate"]
    lr_schedule = [initial_lr, 0.001, 0.0005]
    # Deduplicate while preserving order
    seen = set()
    lr_schedule = [lr for lr in lr_schedule if not (lr in seen or seen.add(lr))]

    last_error: Optional[str] = None
    result = None
    lr_used = None
    lr_retries = 0

    for lr_idx, lr in enumerate(lr_schedule):
        advi_defaults["learning_rate"] = lr
        try:
            advi_result = _run_with_timeout(
                _run_advi_core,
                timeout_seconds,
                Y, p, T, used_guide,
                {**advi_defaults, "low_rank": low_rank},
                rng_seed + lr_idx,
            )
        except Exception as e:
            last_error = repr(e)
            lr_retries += 1
            continue

        # With ``stable_update=True`` inside run_advi, per-step NaN losses
        # are skipped (not propagated) and the state remains valid.  The
        # relevant question is "did at least one restart produce a finite
        # *final* loss?", not "were any intermediate losses NaN?".  We
        # check ``all_final_losses`` (one entry per restart) rather than
        # the full step-level trace.
        all_finals = advi_result.get("all_final_losses", [])
        any_finite_restart = any(math.isfinite(l) for l in all_finals)
        if not any_finite_restart:
            lr_retries += 1
            last_error = f"every restart produced non-finite final loss at lr={lr}"
            continue

        result = advi_result
        lr_used = lr
        break

    if result is None:
        return {
            "status": "failed",
            "error": f"ADVI failed for all learning rates: {last_error}",
            "omega_hat": None,
            "diagnostics": {
                "guide_type": used_guide,
                "guide_requested": guide_type,
                "fallback_to_low_rank": fallback_to_lr,
                "lr_retries": lr_retries,
            },
        }

    # --- Post-process: reconstruct omega/kappa samples from the best guide ---
    samples = result["samples"]
    # Ensure the keys exist; ADVI guides sometimes rename sites.
    if "tau" not in samples:
        return {
            "status": "failed",
            "error": "ADVI posterior samples missing 'tau' site.",
            "omega_hat": None,
            "diagnostics": {"guide_type": used_guide},
        }
    tau, lambdas, omega_diag_samples, omega_offdiag = _reconstruct_samples(samples, p)
    omega_samples = _assemble_omega_matrices(omega_offdiag, omega_diag_samples, p)
    from src.evaluation.shrinkage import compute_kappa_samples
    kappa_samples = compute_kappa_samples(tau, lambdas)
    omega_hat = omega_samples.mean(axis=0)

    losses = np.asarray(result["losses"], dtype=np.float64)
    final_loss = float(losses[-1]) if len(losses) else float("nan")
    all_final_losses = [float(x) for x in result.get("all_final_losses", [])]
    loss_spread_std = (
        float(np.std(all_final_losses)) if len(all_final_losses) > 1 else 0.0
    )

    diagnostics = {
        "guide_type": used_guide,
        "guide_requested": guide_type,
        "fallback_to_low_rank": fallback_to_lr,
        "num_steps": int(advi_defaults["num_steps"]),
        "learning_rate": float(lr_used),
        "lr_retries": int(lr_retries),
        "num_restarts": int(advi_defaults["num_seeds"]),
        "best_seed_index": int(result.get("best_seed", 0)),
        "final_elbo": -final_loss if math.isfinite(final_loss) else None,
        "final_loss": final_loss if math.isfinite(final_loss) else None,
        "all_final_losses": all_final_losses,
        "loss_spread_std": loss_spread_std,
        "num_posterior_samples": int(omega_samples.shape[0]),
        "elapsed_core_seconds": float(result.get("elapsed_seconds", 0.0)),
    }

    return {
        "status": "success",
        "omega_hat": omega_hat,
        "omega_samples": omega_samples,
        "tau_samples": tau,
        "lambda_samples": lambdas,
        "omega_diag_samples": omega_diag_samples,
        "kappa_samples": kappa_samples,
        "elbo_trace": losses,
        "diagnostics": diagnostics,
    }


# ======================================================================
# Method: frequentist
# ======================================================================

def _offdiag_magnitudes(Omega):
    p = Omega.shape[0]
    idx = np.triu_indices(p, k=1)
    return np.abs(Omega[idx])


_GLASSO_ALLOWED_KW = {"alphas", "cv"}


def _run_glasso(Y, p, T, timeout_seconds=None, rng_seed=0, **kwargs):
    """The frequentist runners accept (and ignore) the dispatcher's
    common kwargs (``timeout_seconds``, ``rng_seed``) so the dispatch
    signature stays uniform across all methods.
    """
    from src.benchmarks.frequentist import run_glasso

    glasso_kw = {k: v for k, v in kwargs.items() if k in _GLASSO_ALLOWED_KW}
    Sigma_hat, Omega_hat, alpha = run_glasso(Y, **glasso_kw)
    n_nonzero = int(np.sum(np.abs(Omega_hat) > 0) - p)  # subtract diagonal
    return {
        "status": "success",
        "omega_hat": Omega_hat,
        "sigma_hat": Sigma_hat,
        "offdiag_magnitudes": _offdiag_magnitudes(Omega_hat),
        "diagnostics": {
            "alpha_selected": float(alpha),
            "cv_folds": int(glasso_kw.get("cv", 5)),
            "n_nonzero_offdiag": n_nonzero,
        },
    }


def _run_ledoit_wolf(Y, p, T, timeout_seconds=None, rng_seed=0, **kwargs):
    from src.benchmarks.frequentist import run_ledoit_wolf

    Sigma_hat, Omega_hat, shrinkage = run_ledoit_wolf(Y)
    return {
        "status": "success",
        "omega_hat": Omega_hat,
        "sigma_hat": Sigma_hat,
        "offdiag_magnitudes": _offdiag_magnitudes(Omega_hat),
        "diagnostics": {
            "shrinkage_intensity": float(shrinkage),
        },
    }


def _run_sample_cov(Y, p, T, timeout_seconds=None, rng_seed=0, **kwargs):
    from src.benchmarks.frequentist import run_sample_cov

    Sigma_hat, Omega_hat = run_sample_cov(Y)
    rank = int(np.linalg.matrix_rank(Sigma_hat))
    invertible = Omega_hat is not None

    if not invertible:
        return {
            "status": "singular",
            "omega_hat": None,
            "sigma_hat": Sigma_hat,
            "diagnostics": {
                "rank": rank,
                "invertible": False,
            },
        }
    return {
        "status": "success",
        "omega_hat": Omega_hat,
        "sigma_hat": Sigma_hat,
        "offdiag_magnitudes": _offdiag_magnitudes(Omega_hat),
        "diagnostics": {
            "rank": rank,
            "invertible": True,
        },
    }


# ======================================================================
# Method: Gibbs (Li et al. 2019)
# ======================================================================

def _run_gibbs(Y, p, T, timeout_seconds=None, rng_seed=0, **kwargs):
    """Wrap the Li et al. Gibbs sampler and compute kappa samples."""
    from src.inference.gibbs_runner import run_gibbs
    from src.evaluation.shrinkage import compute_kappa_samples

    gibbs_result = _run_with_timeout(
        run_gibbs,
        timeout_seconds,
        Y=Y,
        p=p,
        rng_seed=rng_seed,
        **{k: v for k, v in kwargs.items()
           if k in ("n_burnin", "n_samples", "n_thinning", "max_rejection")},
    )

    omega_hat = gibbs_result["omega_hat"]
    omega_samples = np.asarray(gibbs_result["omega_samples"])
    tau_sq = np.asarray(gibbs_result["tau_sq_samples"])
    lambda_sq = np.asarray(gibbs_result["lambda_sq_samples"])

    # Convert tau_sq → tau (square root) for kappa computation:
    # kappa = 1 / (1 + lambda^2 * tau^2) = 1 / (1 + lambda_sq * tau_sq)
    kappa_samples = 1.0 / (1.0 + lambda_sq * tau_sq[:, None])

    return {
        "status": "success",
        "omega_hat": omega_hat,
        "omega_samples": omega_samples,
        "tau_samples": np.sqrt(np.maximum(tau_sq, 0)),
        "lambda_samples": np.sqrt(np.maximum(lambda_sq, 0)),
        "omega_diag_samples": np.array([
            np.diag(omega_samples[s]) for s in range(omega_samples.shape[0])
        ]),
        "kappa_samples": kappa_samples,
        "diagnostics": gibbs_result["diagnostics"],
    }


# ======================================================================
# Dispatcher
# ======================================================================

_METHOD_DISPATCH: Dict[str, Callable[..., dict]] = {
    "nuts": _run_nuts,
    "gibbs": _run_gibbs,
    "advi_mf": lambda Y, p, T, **kw: _run_advi(Y, p, T, guide_type="mean_field", **kw),
    "advi_fr": lambda Y, p, T, **kw: _run_advi(Y, p, T, guide_type="full_rank", **kw),
    "advi_lr": lambda Y, p, T, **kw: _run_advi(Y, p, T, guide_type="low_rank", **kw),
    "glasso": _run_glasso,
    "ledoit_wolf": _run_ledoit_wolf,
    "sample_cov": _run_sample_cov,
}


def run_inference(
    method: str,
    data_dir: Path,
    output_dir: Path,
    timeout_seconds: Optional[int] = None,
    rng_seed_offset: int = 0,
    max_saved_samples: int = DEFAULT_MAX_SAVED_SAMPLES,
    **method_kwargs,
) -> dict:
    """Run one inference method on one WORK1 seed directory.

    Parameters
    ----------
    method : str
        One of ``nuts, gibbs, advi_mf, advi_fr, advi_lr, glasso,
        ledoit_wolf, sample_cov``.
    data_dir : Path
        Seed directory from WORK1 containing ``Y.npy``, ``omega_true.npy``,
        ``sigma_true.npy``, and ``metadata.json``.
    output_dir : Path
        Directory where method outputs will be written.  Typically
        ``results/synthetic/<graph>/<p>/<gamma>/<s>/seed_NN/<method>/``.
    timeout_seconds : int, optional
        Linux-only soft timeout.  If the method doesn't finish, a
        ``_Timeout`` exception is raised internally and the run is
        marked ``status="timeout"``.
    rng_seed_offset : int
        Added to the data seed to produce the inference RNG seed.
        Keeps inference RNG deterministic but distinct from data RNG.
    max_saved_samples : int
        Cap on how many posterior samples are saved per file.

    Returns
    -------
    diagnostics : dict
        Always written to ``<output_dir>/diagnostics.json``.  The
        ``status`` key is one of ``success``, ``failed``, ``timeout``,
        ``singular``.
    """
    start_time = time.time()

    if method not in _METHOD_DISPATCH:
        raise ValueError(f"Unknown method: {method!r}")

    # --- 1. Load data ---
    data_dir = Path(data_dir)
    Y = np.load(data_dir / "Y.npy")
    with open(data_dir / "metadata.json") as f:
        data_metadata = json.load(f)

    p = int(data_metadata["p"])
    T = int(data_metadata["T"])
    config_id = int(data_metadata["config_id"])
    seed = int(data_metadata["seed"])
    data_seed = int(data_metadata.get("data_seed", seed))
    rng_seed = data_seed + 1_000_000 + rng_seed_offset  # distinct stream

    base_diagnostics = {
        "method": method,
        "config_id": config_id,
        "seed": seed,
        "p": p,
        "T": T,
        "gamma": data_metadata.get("gamma"),
        "graph": data_metadata.get("graph"),
        "sparsity": data_metadata.get("sparsity"),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "rng_seed": rng_seed,
    }

    # --- 2. Dispatch ---
    fn = _METHOD_DISPATCH[method]
    try:
        result = fn(
            Y, p, T,
            timeout_seconds=timeout_seconds,
            rng_seed=rng_seed,
            **method_kwargs,
        )
    except _Timeout:
        result = {
            "status": "timeout",
            "omega_hat": None,
            "diagnostics": {"timeout_seconds": int(timeout_seconds or 0)},
        }
    except Exception as e:  # defensive — never crash the caller
        result = {
            "status": "failed",
            "error": repr(e),
            "omega_hat": None,
            "diagnostics": {},
        }

    elapsed = time.time() - start_time

    # --- 3. Assemble full diagnostics ---
    diagnostics = {
        **base_diagnostics,
        "status": result.get("status", "unknown"),
        "elapsed_seconds": float(elapsed),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if "error" in result:
        diagnostics["error"] = result["error"]
    diagnostics.update(result.get("diagnostics", {}))

    # --- 4. Save atomically ---
    _save_results_atomic(
        Path(output_dir),
        result,
        diagnostics,
        max_saved_samples=max_saved_samples,
    )

    return diagnostics
