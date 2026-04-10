"""ADVI (Automatic Differentiation Variational Inference) runner."""

import jax


def run_advi(
    model,
    Y,
    p,
    guide_type="mean_field",
    num_steps=50000,
    learning_rate=0.01,
    num_samples=5000,
    num_seeds=5,
    rng_seed=0,
):
    """Run ADVI on the graphical horseshoe model.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    Y : jnp.ndarray, shape (T, p)
        Observation matrix.
    p : int
        Dimension.
    guide_type : str
        One of "mean_field" (AutoNormal), "full_rank" (AutoMultivariateNormal),
        "low_rank" (AutoLowRankMultivariateNormal), or "map" (AutoDelta).
    num_steps : int
        Number of optimization steps.
    learning_rate : float
        Initial learning rate for Adam optimizer.
    num_samples : int
        Number of samples to draw from the fitted guide.
    num_seeds : int
        Number of random restarts to assess initialization sensitivity.
    rng_seed : int
        Random seed.

    Returns
    -------
    svi_result : dict
        Dictionary with keys: "samples", "losses", "guide_params", "best_seed".
    """
    raise NotImplementedError
