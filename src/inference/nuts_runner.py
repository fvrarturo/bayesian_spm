"""NUTS (No-U-Turn Sampler) inference runner for the graphical horseshoe."""

import time

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS


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
):
    """Run NUTS on the graphical horseshoe model.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    Y : jnp.ndarray, shape (T, p)
        Observation matrix.
    p : int
        Dimension.
    num_warmup : int
        Number of warmup (adaptation) steps per chain.
    num_samples : int
        Number of posterior samples per chain.
    num_chains : int
        Number of independent chains.
    target_accept_prob : float
        Target acceptance probability for step size adaptation.
    max_tree_depth : int
        Maximum tree depth for NUTS.
    rng_seed : int
        Random seed.

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object with posterior samples.
    """
    Y_jnp = jnp.array(Y)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(rng_seed)
    start = time.time()
    mcmc.run(rng_key, Y=Y_jnp, p=p)
    elapsed = time.time() - start

    mcmc.print_summary()
    print(f"\nNUTS wall-clock time: {elapsed:.1f}s")

    return mcmc


def extract_omega_samples(mcmc, p):
    """Extract assembled precision matrix samples from MCMC output.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object.
    p : int
        Dimension.

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
    n_samples = offdiag.shape[0]
    idx_upper = jnp.triu_indices(p, k=1)

    def assemble_one(offdiag_i, diag_i):
        Omega = jnp.zeros((p, p))
        Omega = Omega.at[idx_upper].set(offdiag_i)
        Omega = Omega + Omega.T + jnp.diag(diag_i)
        return Omega

    Omega_samples = jax.vmap(assemble_one)(offdiag, diag)
    return Omega_samples
