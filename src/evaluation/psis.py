"""Pareto Smoothed Importance Sampling (PSIS) diagnostic for VI quality.

Measures how well the variational approximation q(θ) matches the
true posterior p(θ|y) by importance-weighting VI samples:

    w_s = p(y, θ_s) / q(θ_s)    for θ_s ~ q

The tail shape parameter k̂ of the (log) importance weights indicates:

    k̂ < 0.5   →  q is a good approximation
    0.5–0.7   →  marginal; results may be unreliable
    k̂ > 0.7   →  q is a poor approximation

References
----------
Vehtari, Gelman, Gabry (2017).  Pareto smoothed importance sampling.
Yao, Vehtari, Simpson, Gelman (2018).  Yes, but did it work?
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ======================================================================
# Core khat estimator
# ======================================================================

def _fit_gpd_khat(log_weights: np.ndarray) -> float:
    """Estimate the GPD shape parameter k̂ from log importance weights.

    Uses the Zhang & Stephens (2009) method-of-moments estimator on the
    upper tail of the sorted log-weights.  Falls back to arviz if
    available.
    """
    # Try arviz first (gold standard implementation)
    try:
        import arviz as az
        # arviz >= 0.12 API
        if hasattr(az, "psislw"):
            lw, khat = az.psislw(log_weights.ravel())
            return float(np.asarray(khat).ravel()[0])
    except Exception:
        pass

    # Fallback: simple moment estimator on the upper tail
    lw = np.asarray(log_weights, dtype=np.float64).ravel()
    n = lw.size
    if n < 10:
        return float("nan")

    # Number of tail samples (Vehtari et al. recommend min(n/5, 3√n))
    M = int(min(n * 0.2, 3.0 * np.sqrt(n)))
    M = max(M, 5)

    sorted_lw = np.sort(lw)[::-1]  # descending
    threshold = sorted_lw[M]
    excesses = sorted_lw[:M] - threshold

    if excesses.max() <= 0:
        return 0.0  # all weights are equal → perfect match

    # Method-of-moments estimator for GPD shape k:
    #   E[X] = σ/(1-k),  Var[X] = σ²/((1-k)²(1-2k))
    #   → k = 0.5*(1 - mean²/var)
    mean_e = np.mean(excesses)
    var_e = np.var(excesses, ddof=1)

    if mean_e <= 0 or var_e <= 0:
        return 0.0
    khat = 0.5 * (1.0 - mean_e ** 2 / var_e)
    return float(np.clip(khat, -0.5, 2.0))  # sensible bounds


def interpret_khat(khat: float) -> str:
    """Human-readable interpretation of the PSIS k̂ value."""
    if np.isnan(khat):
        return "unknown"
    if khat < 0.5:
        return "good"
    if khat < 0.7:
        return "marginal"
    return "bad"


# ======================================================================
# Log-weight computation via NumPyro Trace_ELBO
# ======================================================================

def compute_psis_khat_from_svi(
    model,
    guide,
    guide_params,
    Y,
    p: int,
    n_eval_samples: int = 200,
    rng_seed: int = 99,
) -> dict:
    """Compute PSIS k̂ for a fitted ADVI guide.

    Uses NumPyro's ``Trace_ELBO.loss`` to compute per-sample ELBO
    values, which equal the (negative) log importance weights
    ``log w_s = log p(y, θ_s) - log q(θ_s)``.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    guide : AutoGuide instance (already built, not just a class).
    guide_params : dict
        Fitted guide parameters (from ``svi_result.params``).
    Y : array-like, shape (T, p)
    p : int
    n_eval_samples : int
        Number of importance weights to compute.  200 is typically
        sufficient for a stable k̂ estimate.
    rng_seed : int

    Returns
    -------
    dict with ``psis_khat``, ``psis_interpretation``, ``n_eval_samples``.
    """
    try:
        import jax
        import jax.numpy as jnp
        from numpyro.infer import Trace_ELBO, SVI
        from numpyro.optim import Adam

        Y_jnp = jnp.asarray(Y)
        elbo_fn = Trace_ELBO(num_particles=1)

        # We need a dummy SVI object to access the loss method.
        # The optimizer doesn't matter — we only evaluate, never step.
        dummy_opt = Adam(1e-3)
        svi = SVI(model, guide, dummy_opt, loss=elbo_fn)

        log_weights = []
        rng_key = jax.random.PRNGKey(rng_seed)
        for i in range(n_eval_samples):
            rng_i = jax.random.fold_in(rng_key, i)
            # loss = -ELBO = -(log p(y,θ) - log q(θ)), so log_w = -loss
            svi_state = svi.init(rng_i, Y=Y_jnp, p=p)
            # Overwrite params with the fitted ones
            svi_state = svi_state._replace(optim_state=svi.optim.init(guide_params))
            loss_val = svi.evaluate(svi_state, Y=Y_jnp, p=p)
            log_weights.append(-float(loss_val))

        log_weights = np.array(log_weights)
        khat = _fit_gpd_khat(log_weights)
    except Exception as e:
        return {
            "psis_khat": None,
            "psis_interpretation": "error",
            "psis_error": repr(e),
            "n_eval_samples": 0,
        }

    return {
        "psis_khat": float(khat) if np.isfinite(khat) else None,
        "psis_interpretation": interpret_khat(khat),
        "n_eval_samples": int(n_eval_samples),
    }


# ======================================================================
# Standalone khat from pre-computed log-weights
# ======================================================================

def compute_psis_khat(log_weights: np.ndarray) -> dict:
    """Compute PSIS k̂ from an array of log importance weights.

    Parameters
    ----------
    log_weights : np.ndarray, shape (n,)

    Returns
    -------
    dict with ``psis_khat``, ``psis_interpretation``.
    """
    lw = np.asarray(log_weights, dtype=np.float64).ravel()
    finite = lw[np.isfinite(lw)]
    if len(finite) < 10:
        return {"psis_khat": None, "psis_interpretation": "insufficient_samples"}
    khat = _fit_gpd_khat(finite)
    return {
        "psis_khat": float(khat) if np.isfinite(khat) else None,
        "psis_interpretation": interpret_khat(khat),
    }
