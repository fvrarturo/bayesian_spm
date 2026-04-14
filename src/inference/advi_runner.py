"""ADVI (Automatic Differentiation Variational Inference) runner.

Wraps NumPyro's SVI with a NumPyro AutoGuide, running a multi-seed
restart protocol and keeping the guide with the best final ELBO.  The
guide is initialised via ``init_to_median`` by default — the NumPyro
default ``init_to_uniform`` produces a non-PD Omega on the first pass
for the graphical horseshoe model (the log-density is NaN everywhere
on random starts) and ADVI immediately fails with "Cannot find valid
initial parameters".
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)

# ``ClippedAdam`` clips gradients elementwise before the Adam update,
# preventing a single outlier sample from blowing up variational
# parameters on the horseshoe's heavy-tailed prior.  Fall back to
# plain Adam on very old NumPyro installs.
try:
    from numpyro.optim import ClippedAdam as _AdamOptimizer
    _CLIP_NORM = 10.0
    _USE_CLIP = True
except ImportError:  # pragma: no cover
    from numpyro.optim import Adam as _AdamOptimizer
    _CLIP_NORM = None
    _USE_CLIP = False


def _make_optimizer(learning_rate):
    if _USE_CLIP:
        return _AdamOptimizer(step_size=learning_rate, clip_norm=_CLIP_NORM)
    return _AdamOptimizer(step_size=learning_rate)

# Import ``init_to_median`` and ``init_to_value`` with a fallback chain
# for older NumPyro installs.  Matches the fallback used in ``nuts_runner``.
try:
    from numpyro.infer import init_to_median, init_to_value  # NumPyro >= 0.7
except ImportError:  # pragma: no cover
    try:
        from numpyro.infer.initialization import init_to_median, init_to_value
    except ImportError:  # pragma: no cover
        from numpyro.infer.util import init_to_median, init_to_value


GUIDE_MAP = {
    "mean_field": AutoNormal,
    "full_rank": AutoMultivariateNormal,
    "low_rank": AutoLowRankMultivariateNormal,
    "map": AutoDelta,
}


def _build_guide(guide_cls, model, init_loc_fn, init_scale=0.01, low_rank=None):
    """Construct the AutoGuide, passing guide-specific extras.

    ``init_scale`` controls the initial variance of the variational
    distribution.  NumPyro's default is 0.1, but for the graphical
    horseshoe this lets ADVI draw samples whose implied Omega is
    non-PD, producing NaN log-densities that wreck the gradient.
    0.01 gives a tighter initial distribution; ADVI then widens it
    during optimization if the posterior supports it.
    """
    kwargs = {"init_loc_fn": init_loc_fn}
    if guide_cls is not AutoDelta:
        kwargs["init_scale"] = init_scale
    if guide_cls is AutoLowRankMultivariateNormal and low_rank is not None:
        kwargs["rank"] = int(low_rank)
    return guide_cls(model, **kwargs)


def run_advi(
    model,
    Y,
    p,
    guide_type="mean_field",
    num_steps=50000,
    learning_rate=0.005,
    num_samples=5000,
    num_seeds=5,
    rng_seed=0,
    low_rank=None,
    init_scale=0.01,
    num_particles=1,
    init_values=None,
):
    """Run ADVI on the graphical horseshoe model.

    Uses NumPyro's fast ``svi.run(..., stable_update=True)`` path, which
    compiles the entire optimization loop as a single ``jax.lax.scan``
    and automatically skips any gradient step that produces a non-finite
    loss.  This is orders of magnitude faster than a Python for-loop
    calling ``svi.update`` per iteration (the earlier design of this
    module), especially on p>=50 where per-step Python/device-sync
    overhead became the dominant cost.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    Y : array-like, shape (T, p)
    p : int
    guide_type : str
        One of ``mean_field`` (AutoNormal), ``full_rank``
        (AutoMultivariateNormal), ``low_rank``
        (AutoLowRankMultivariateNormal), or ``map`` (AutoDelta).
    num_steps : int
        Gradient steps per random restart.
    learning_rate : float
        Adam learning rate.
    num_samples : int
        Posterior samples drawn from the best guide.
    num_seeds : int
        Number of random restarts.  The one with the lowest final loss
        (= highest ELBO) is kept.
    rng_seed : int
        Base seed; each restart uses ``rng_seed + seed_idx``.
    low_rank : int, optional
        Rank for ``AutoLowRankMultivariateNormal``.  Ignored for other
        guide types.
    init_scale : float
        Initial variational scale.  Small values (0.01) keep samples
        close to the median on the horseshoe and avoid NaN log-densities.
    num_particles : int
        Monte Carlo samples per gradient step.  Default 1 (NumPyro's
        default).  Higher values reduce gradient variance but multiply
        per-step cost linearly.

    Returns
    -------
    dict with keys ``samples, losses, guide_params, best_seed,
    all_final_losses, elapsed_seconds``.
    """
    if guide_type not in GUIDE_MAP:
        raise ValueError(
            f"Unknown guide_type: {guide_type}. Choose from {list(GUIDE_MAP)}"
        )

    Y_jnp = jnp.asarray(Y)
    guide_cls = GUIDE_MAP[guide_type]

    best_loss = float("inf")
    best_params = None
    best_guide = None
    best_seed_idx = 0
    best_losses = None
    all_final_losses = []

    start = time.time()

    for seed_idx in range(num_seeds):
        rng_key = jax.random.PRNGKey(rng_seed + seed_idx)

        # Fresh guide per restart.
        # When ``init_values`` is supplied (recommended for high-p), use
        # ``init_to_value`` with hand-picked PD-safe starting parameters.
        # Otherwise fall back to ``init_to_median(num_samples=15)`` which
        # works for p≤50 but can fail at p=100.
        if init_values is not None:
            init_loc = init_to_value(values=init_values)
        else:
            init_loc = init_to_median(num_samples=15)

        guide = _build_guide(
            guide_cls,
            model,
            init_loc_fn=init_loc,
            init_scale=init_scale,
            low_rank=low_rank,
        )

        optimizer = _make_optimizer(learning_rate)
        svi = SVI(
            model,
            guide,
            optimizer,
            loss=Trace_ELBO(num_particles=num_particles),
        )

        print(
            f"  Seed {seed_idx}: running {num_steps} steps "
            f"(guide={guide_type}, lr={learning_rate}, "
            f"num_particles={num_particles})...",
            flush=True,
        )
        seed_start = time.time()
        svi_result = svi.run(
            rng_key,
            num_steps,
            Y=Y_jnp,
            p=p,
            progress_bar=False,
            stable_update=True,
        )
        seed_elapsed = time.time() - seed_start

        losses = np.asarray(svi_result.losses)
        # stable_update may leave the final loss finite even if early
        # steps were NaN-skipped; also guard against a fully-NaN run.
        finite_mask = np.isfinite(losses)
        if not finite_mask.any():
            final_loss = float("nan")
        else:
            final_loss = float(losses[np.where(finite_mask)[0][-1]])

        all_final_losses.append(final_loss)
        print(
            f"  Seed {seed_idx}: done in {seed_elapsed:.1f}s, "
            f"final loss={final_loss:.2f}, "
            f"finite steps={int(finite_mask.sum())}/{len(losses)}",
            flush=True,
        )

        if math.isfinite(final_loss) and final_loss < best_loss:
            best_loss = final_loss
            best_params = svi_result.params
            best_guide = guide
            best_seed_idx = seed_idx
            best_losses = losses.tolist()

    elapsed = time.time() - start

    if best_guide is None:
        # Every seed produced a non-finite final loss — propagate.
        return {
            "samples": None,
            "losses": [],
            "guide_params": None,
            "best_seed": -1,
            "all_final_losses": all_final_losses,
            "elapsed_seconds": elapsed,
        }

    # --- Draw posterior samples from the best guide ---
    rng_key = jax.random.PRNGKey(rng_seed + 1000)
    predictive = numpyro.infer.Predictive(
        best_guide, params=best_params, num_samples=num_samples
    )
    samples = predictive(rng_key, Y=Y_jnp, p=p)

    print(
        f"\nADVI ({guide_type}) done. Best seed: {best_seed_idx}, "
        f"final loss: {best_loss:.2f}, wall-clock: {elapsed:.1f}s",
        flush=True,
    )

    return {
        "samples": samples,
        "losses": best_losses or [],
        "guide_params": best_params,
        "best_seed": best_seed_idx,
        "all_final_losses": all_final_losses,
        "elapsed_seconds": elapsed,
    }
