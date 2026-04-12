"""NUTS (No-U-Turn Sampler) inference runner for the graphical horseshoe.

This module exposes two functions:

- ``run_nuts(model, Y, p, ...)`` — runs NUTS with the given hyperparameters
  and returns the fitted ``numpyro.infer.MCMC`` object.  ``extra_fields``
  defaults to ``("diverging",)`` so the caller can count divergent
  transitions for convergence diagnostics.
- ``extract_omega_samples(mcmc, p)`` — reassembles a 3D stack of full
  precision matrices from the per-parameter posterior samples, via JAX
  ``vmap``.

Designed to be a pure building block.  Higher-level concerns (atomic
saves, retries, timeouts, divergence-triggered reparameterization) live
in ``src/inference/run_single.py``.
"""

import time

import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS

# ``init_to_median`` lives in different submodules across NumPyro versions.
# Try the top-level ``numpyro.infer`` first, then fall back to the internal
# ``numpyro.infer.initialization`` module used by older releases.
try:
    from numpyro.infer import init_to_median  # NumPyro >= 0.7
except ImportError:  # pragma: no cover
    try:
        from numpyro.infer.initialization import init_to_median
    except ImportError:  # pragma: no cover
        from numpyro.infer.util import init_to_median  # very old fallback


def run_nuts(
    model,
    Y,
    p,
    num_warmup=2000,
    num_samples=5000,
    num_chains=4,
    target_accept_prob=0.85,
    max_tree_depth=10,
    rng_seed=0,
    progress_bar=False,
    extra_fields=("diverging",),
    init_strategy=None,
):
    """Run NUTS on a NumPyro model.

    Parameters
    ----------
    model : callable
        NumPyro model function.  Must accept ``Y`` and ``p`` as keyword
        arguments (any other model-level options should be baked in via
        closure before passing in).
    Y : array-like, shape (T, p)
    p : int
    num_warmup : int
    num_samples : int
    num_chains : int
    target_accept_prob : float
    max_tree_depth : int
    rng_seed : int
    progress_bar : bool
        Pass ``False`` on cluster nodes where stdout is captured.
    extra_fields : tuple of str
        Transition fields to record.  ``"diverging"`` must be in this
        tuple if you want to count divergent transitions afterward.

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        The fitted MCMC object.  Use ``mcmc.get_samples()`` for the
        posterior samples, ``mcmc.get_samples(group_by_chain=True)`` plus
        ``numpyro.diagnostics.summary`` for per-parameter R-hat and ESS,
        and ``mcmc.get_extra_fields()["diverging"]`` for divergence
        indicators.
    """
    Y_jnp = jnp.asarray(Y)

    # Default to init_to_median: starts at prior medians where z = 0,
    # which makes the initial Omega diagonal (= diag(omega_diag)) and
    # therefore trivially positive definite.  The NumPyro default
    # (init_to_uniform) draws z, log(tau), log(lambda) ~ U(-2, 2), which
    # gives |omega_offdiag| up to ~100 while omega_diag is at most ~7,
    # producing a wildly indefinite Omega on the first step and causing
    # every leapfrog trajectory to diverge with NaN log-density.
    if init_strategy is None:
        init_strategy = init_to_median(num_samples=15)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(rng_key, Y=Y_jnp, p=p, extra_fields=extra_fields)
    return mcmc


def extract_omega_samples(mcmc, p):
    """Reassemble full precision matrices from MCMC samples.

    Handles both parameterizations transparently:
    - If ``omega_offdiag`` is present as a deterministic site (the
      non-centered default), it is used directly.
    - Otherwise the centered offdiag samples are reconstructed from
      ``z * lambdas * tau``.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
    p : int

    Returns
    -------
    Omega_samples : jnp.ndarray, shape (n_samples, p, p)
    """
    samples = mcmc.get_samples()

    if "omega_offdiag" in samples:
        offdiag = samples["omega_offdiag"]
    else:
        z = samples["z"]
        lambdas = samples["lambdas"]
        tau = samples["tau"]
        offdiag = z * lambdas * tau[:, None]

    diag = samples["omega_diag"]
    idx_upper = jnp.triu_indices(p, k=1)

    def assemble_one(offdiag_i, diag_i):
        Omega = jnp.zeros((p, p))
        Omega = Omega.at[idx_upper].set(offdiag_i)
        Omega = Omega + Omega.T + jnp.diag(diag_i)
        return Omega

    return jax.vmap(assemble_one)(offdiag, diag)
