"""NUTS (No-U-Turn Sampler) inference runner for the graphical horseshoe."""

import jax


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
    raise NotImplementedError
