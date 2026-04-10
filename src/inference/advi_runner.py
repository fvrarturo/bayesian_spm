"""ADVI (Automatic Differentiation Variational Inference) runner."""

import time

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from numpyro.optim import Adam


GUIDE_MAP = {
    "mean_field": AutoNormal,
    "full_rank": AutoMultivariateNormal,
    "low_rank": AutoLowRankMultivariateNormal,
    "map": AutoDelta,
}


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
        Dictionary with keys: "samples", "losses", "guide_params", "best_seed",
        "all_final_losses", "elapsed_seconds".
    """
    if guide_type not in GUIDE_MAP:
        raise ValueError(f"Unknown guide_type: {guide_type}. Choose from {list(GUIDE_MAP)}")

    Y_jnp = jnp.array(Y)

    best_loss = float("inf")
    best_params = None
    best_guide = None
    best_seed_idx = 0
    all_final_losses = []
    all_losses = []

    start = time.time()

    for seed_idx in range(num_seeds):
        rng_key = jax.random.PRNGKey(rng_seed + seed_idx)

        guide_cls = GUIDE_MAP[guide_type]
        guide = guide_cls(model)

        optimizer = Adam(learning_rate)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(rng_key, Y=Y_jnp, p=p)

        losses = []
        for step in range(num_steps):
            svi_state, loss = svi.update(svi_state, Y=Y_jnp, p=p)
            losses.append(float(loss))

            if (step + 1) % 10000 == 0:
                print(f"  Seed {seed_idx}, step {step+1}/{num_steps}, ELBO={-loss:.2f}")

        final_loss = losses[-1]
        all_final_losses.append(final_loss)

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = svi.get_params(svi_state)
            best_guide = guide
            best_seed_idx = seed_idx
            all_losses = losses

    elapsed = time.time() - start

    # Draw posterior samples from the best guide
    rng_key = jax.random.PRNGKey(rng_seed + 1000)
    predictive = numpyro.infer.Predictive(
        best_guide, params=best_params, num_samples=num_samples
    )
    samples = predictive(rng_key, Y=Y_jnp, p=p)

    print(f"\nADVI ({guide_type}) done. Best seed: {best_seed_idx}, "
          f"final loss: {best_loss:.2f}, wall-clock: {elapsed:.1f}s")

    return {
        "samples": samples,
        "losses": all_losses,
        "guide_params": best_params,
        "best_seed": best_seed_idx,
        "all_final_losses": all_final_losses,
        "elapsed_seconds": elapsed,
    }
