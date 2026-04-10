"""Graphical horseshoe model for sparse precision matrix estimation.

Implements the model from Li, Craig, and Bhadra (2019) in NumPyro.
Horseshoe priors on off-diagonal elements of Omega with half-Cauchy
local and global shrinkage parameters.

Model (non-centered parameterization):
    tau        ~ HalfCauchy(0, tau_scale)
    lambda_ij  ~ HalfCauchy(0, 1)           for i < j
    z_ij       ~ Normal(0, 1)               for i < j
    omega_ij   = z_ij * lambda_ij * tau      for i < j
    omega_ii   ~ HalfNormal(5)               (or Exponential / Gamma)
    Omega      = assemble(omega_ij, omega_ii), Omega in S_p^+
    Y_k        ~ N(0, Omega^{-1})            for k = 1,...,T
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def graphical_horseshoe(Y, p, ncp=True, tau_scale=1.0, diag_prior="halfnormal"):
    """Graphical horseshoe model for precision matrix estimation.

    Parameters
    ----------
    Y : jnp.ndarray, shape (T, p)
        Zero-mean observation matrix.
    p : int
        Dimension (number of variables).
    ncp : bool
        If True, use non-centered parameterization (recommended for NUTS).
    tau_scale : float
        Scale of the half-Cauchy prior on the global shrinkage tau.
    diag_prior : str
        Prior for diagonal elements: "halfnormal", "exponential", or "gamma".
    """
    n_offdiag = p * (p - 1) // 2

    # --- Global shrinkage ---
    tau = numpyro.sample("tau", dist.HalfCauchy(tau_scale))

    # --- Local shrinkage ---
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(n_offdiag)))

    # --- Off-diagonal elements ---
    if ncp:
        z = numpyro.sample("z", dist.Normal(jnp.zeros(n_offdiag), 1.0))
        omega_offdiag = numpyro.deterministic("omega_offdiag", z * lambdas * tau)
    else:
        omega_offdiag = numpyro.sample(
            "omega_offdiag", dist.Normal(0, lambdas * tau)
        )

    # --- Diagonal elements ---
    if diag_prior == "halfnormal":
        omega_diag = numpyro.sample(
            "omega_diag", dist.HalfNormal(jnp.ones(p) * 5.0)
        )
    elif diag_prior == "exponential":
        omega_diag = numpyro.sample(
            "omega_diag", dist.Exponential(jnp.ones(p) * 0.5)
        )
    elif diag_prior == "gamma":
        omega_diag = numpyro.sample(
            "omega_diag", dist.Gamma(2.0 * jnp.ones(p), 0.5 * jnp.ones(p))
        )
    else:
        raise ValueError(f"Unknown diag_prior: {diag_prior}")

    # --- Assemble symmetric precision matrix ---
    Omega = jnp.zeros((p, p))
    idx_upper = jnp.triu_indices(p, k=1)
    Omega = Omega.at[idx_upper].set(omega_offdiag)
    Omega = Omega + Omega.T
    Omega = Omega + jnp.diag(omega_diag)

    # --- Likelihood ---
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(p), precision_matrix=Omega),
        obs=Y,
    )
