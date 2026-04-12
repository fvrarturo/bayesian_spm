# Agent Prompt: WORK2 — Inference & Evaluation Pipeline

## Context

You are building the **inference and evaluation pipeline** for a research project comparing MCMC (NUTS) vs. Variational Inference (ADVI) for sparse Bayesian precision matrix estimation using the graphical horseshoe prior. This is for MIT course 6.7830 (Bayesian Modeling and Inference, Spring 2026). The progress report is due **April 15** (4 days from now); the final report is due **May 5**.

**WORK1 is complete.** The synthetic data generation pipeline produced 1,680 validated (Ω₀, Σ₀, Y, metadata) triples across 84 configurations × 20 seeds. The data lives at `data/synthetic/` in the directory structure described below. Every file has been audited: 0 missing, 0 corrupted, 55 unit tests pass, 8 distributional sanity checks pass.

The codebase already has partial implementations of the inference modules (see §3 for what exists and what needs to be finished). Your job is to:

1. **Validate** the existing NumPyro model on a trivial case (smoke test gate).
2. **Build** a complete inference runner that loads a pre-generated seed directory and runs one method.
3. **Build** an evaluation module that computes all metrics given an inference result and the ground truth.
4. **Build** the SLURM orchestration to run all methods across the experimental grid.
5. **Produce preliminary results** for the progress report (narrow slice of the grid).

---

## 1. What Already Exists

### 1.1 Data Layout (from WORK1)

```
data/synthetic/
  configs/
    config_manifest.json              # 84 configs, source of truth
  erdos_renyi/
    p050/gamma042/s010/seed_00/
      omega_true.npy                  # (p, p) float64
      sigma_true.npy                  # (p, p) float64
      Y.npy                           # (T, p) float64
      metadata.json                   # scalars + edge_set + warnings
  block_diagonal/
    ...
```

Each `metadata.json` contains `config_id`, `p`, `T`, `gamma`, `graph`, `sparsity`, `seed`, `n_edges`, `edge_set` (as list of [i,j] pairs), `condition_number`, `eigenvalues` (from Ω₀), `oracle_portfolio_variance`, and more. The `config_manifest.json` maps `config_id` → full config dict including `dir_path`.

### 1.2 Existing Code Modules

| Module | Status | What's there |
|--------|--------|-------------|
| `src/utils/matrix_utils.py` | ✅ Complete | `assemble_precision_matrix`, all generators, `sample_data_from_omega` |
| `src/utils/validation.py` | ✅ Complete | `validate_omega`, `validate_sigma`, `validate_data` |
| `src/utils/configs.py` | ✅ Complete | `compute_configs`, directory name helpers, `T_from_gamma` |
| `src/models/graphical_horseshoe.py` | 🟡 Exists, needs validation | `graphical_horseshoe(Y, p, ncp=True, tau_scale=1.0, diag_prior="halfnormal")` |
| `src/inference/nuts_runner.py` | 🟡 Exists, needs integration | `run_nuts(model, Y, p, ...)`, `extract_omega_samples(mcmc, p)` |
| `src/inference/advi_runner.py` | 🟡 Exists, needs integration | `run_advi(model, Y, p, ...)`, multi-seed protocol |
| `src/benchmarks/frequentist.py` | 🟡 Exists, needs integration | `run_sample_cov`, `run_ledoit_wolf`, `run_glasso` |
| `src/evaluation/metrics.py` | 🟡 Exists, needs completion | `steins_loss`, `frobenius_loss`, `sparsity_metrics` (partial) |
| `src/portfolio/gmv.py` | 🟡 Exists, needs integration | `gmv_weights`, `portfolio_variance` |
| `src/utils/plotting.py` | 🟡 Stubs exist | `plot_precision_heatmap`, `plot_shrinkage_profile`, etc. |

### 1.3 The NumPyro Model (already written)

The model in `src/models/graphical_horseshoe.py` implements the generative process (non-centered parameterization by default):

```
τ ~ C⁺(0, τ_scale)
λ_ij ~ C⁺(0, 1)           for all i < j
z_ij ~ N(0, 1)             for all i < j
ω_ij = z_ij · λ_ij · τ     (deterministic, non-centered)
ω_ii ~ HalfNormal(5)       for all i
Ω = assemble(ω_offdiag, ω_diag)
Y_k ~ N(0, Ω⁻¹)           for k = 1, ..., T
```

The model uses JAX's `.at[].set()` for differentiable assembly and NumPyro's `MultivariateNormal(precision_matrix=Omega)` for the likelihood. PD is enforced implicitly: non-PD proposals get log-density −∞ and NUTS rejects them.

---

## 2. The Six Methods to Run

### 2.1 Method Specifications

| Method ID | Full Name | Implementation | Key Parameters | Expected Runtime (p=50) |
|-----------|-----------|---------------|----------------|------------------------|
| `nuts` | NUTS (No-U-Turn Sampler) | NumPyro MCMC | num_warmup=2000, num_samples=5000, num_chains=4, target_accept=0.85, max_tree_depth=10 | 30–120 min/seed |
| `advi_mf` | ADVI Mean-Field | NumPyro SVI + AutoNormal | num_steps=50000, lr=0.01, num_seeds=5 | 5–15 min/seed |
| `advi_fr` | ADVI Full-Rank | NumPyro SVI + AutoMultivariateNormal | num_steps=100000, lr=0.005, num_seeds=3 | 15–60 min/seed |
| `glasso` | Graphical Lasso (CV) | sklearn GraphicalLassoCV | cv=5, max_iter=500 | < 10 sec/seed |
| `ledoit_wolf` | Ledoit–Wolf Shrinkage | sklearn LedoitWolf | (analytical, no tuning) | < 1 sec/seed |
| `sample_cov` | Sample Covariance | numpy | None | < 1 sec/seed |

### 2.2 Method-Specific Details

#### NUTS

- Run **4 independent chains** for convergence diagnostics (R̂, ESS).
- Use **non-centered parameterization** (ncp=True) as default. If divergences > 10% of samples, retry with centered (ncp=False) and log the switch.
- NUTS at **p=100 may be infeasible**. Set a wall-time budget of 4 hours per (config, seed). If it doesn't finish, record `status: "timeout"` and move on. This is an expected finding, not a bug.
- For p=10: reduce warmup to 1000 and samples to 2000 (the posterior is simple, no need to waste compute).
- Save: full posterior samples for Ω (shape: `(n_samples, p, p)`), plus τ samples, λ samples, and per-parameter R̂ and ESS.

#### ADVI Mean-Field

- Guide: `AutoNormal(graphical_horseshoe)` — independent Gaussian per latent in unconstrained space.
- Run from **5 different random seeds** (initialization sensitivity is a key diagnostic for the horseshoe).
- Keep the run with the **lowest final ELBO loss** (= highest ELBO).
- From the best guide, draw **5000 posterior samples** via `numpyro.infer.Predictive`.
- Save: posterior samples (same shape as NUTS), ELBO trace (per-iteration losses for the best seed), all 5 final losses (to quantify initialization sensitivity), and wall-clock time.
- If the loss is `nan` or `inf` at any point, reduce learning rate to 0.005 and retry. If it still diverges, reduce to 0.001. Log all retries.

#### ADVI Full-Rank

- Guide: `AutoMultivariateNormal(graphical_horseshoe)`.
- **Feasibility check**: for p=50, D ≈ 2501 latent dimensions. The Cholesky factor of the guide covariance has D(D+1)/2 ≈ 3.1M parameters. This is large but likely feasible with patience (100k steps).
- For **p=100**, D ≈ 10001, Cholesky has ~50M parameters — almost certainly **infeasible**. Fall back to `AutoLowRankMultivariateNormal(graphical_horseshoe, rank=50)` and record this as `method: "advi_lr"` with a note. If even low-rank fails, record `status: "failed"` and move on.
- Run from **3 random seeds** (fewer than MF because each run is more expensive).
- Lower learning rate: `lr=0.005` (full-rank is less stable).
- Save: same as ADVI-MF, plus the guide type actually used (full-rank vs low-rank).

#### Graphical Lasso

- Use `sklearn.covariance.GraphicalLassoCV(cv=5, assume_centered=True, max_iter=500)`.
- The data Y is already zero-mean by construction, so `assume_centered=True` is correct.
- Save: point estimate Ω̂, selected regularization parameter α, and wall-clock time.
- **No posterior samples** — this is a point estimator.

#### Ledoit–Wolf

- Use `sklearn.covariance.LedoitWolf(assume_centered=True)`.
- Produces Σ̂_LW; invert to get Ω̂_LW = Σ̂⁻¹_LW.
- Save: Ω̂, Σ̂, shrinkage intensity α̂, and wall-clock time.
- **Note**: Ledoit–Wolf does NOT produce a sparse Ω̂. All off-diagonal entries will be nonzero. This is expected — it's a non-sparse benchmark showing what you get without sparsity assumptions.

#### Sample Covariance

- Compute Σ̂ = (T−1)⁻¹ YᵀY via `numpy.cov(Y, rowvar=False, bias=False)`.
- If T > p, invert to get Ω̂. If T ≤ p, the sample covariance is rank-deficient and Ω̂ **does not exist**. Record `status: "singular"` and `omega_hat: null`.
- This is a **baseline showing how bad things get without regularization**. It should have the worst metrics at high γ.

---

## 3. What Each Inference Run Produces

Every inference run — regardless of method — writes its outputs into a **results directory** that mirrors the data directory structure. This keeps data and results cleanly separated.

### 3.1 Output Directory Structure

```
results/
  synthetic/
    erdos_renyi/
      p050/gamma042/s010/seed_00/
        nuts/
          omega_hat.npy               # (p, p) — posterior mean
          omega_samples.npy           # (n_samples, p, p) — full posterior [Bayesian only]
          tau_samples.npy             # (n_samples,) [Bayesian only]
          lambda_samples.npy          # (n_samples, n_offdiag) [Bayesian only]
          kappa_samples.npy           # (n_samples, n_offdiag) — shrinkage coefficients [Bayesian only]
          diagnostics.json            # method-specific diagnostics
          metrics.json                # evaluation metrics against ground truth
        advi_mf/
          omega_hat.npy
          omega_samples.npy
          tau_samples.npy
          lambda_samples.npy
          kappa_samples.npy
          elbo_trace.npy              # (num_steps,) — per-iteration negative ELBO [ADVI only]
          diagnostics.json
          metrics.json
        advi_fr/
          ...
        glasso/
          omega_hat.npy
          diagnostics.json
          metrics.json
        ledoit_wolf/
          omega_hat.npy
          diagnostics.json
          metrics.json
        sample_cov/
          omega_hat.npy               # may be null/absent if T ≤ p
          diagnostics.json
          metrics.json
```

### 3.2 The `diagnostics.json` Schema

Every method writes a `diagnostics.json` with at minimum:

```json
{
  "method": "nuts",
  "config_id": 7,
  "seed": 3,
  "p": 50,
  "T": 120,
  "status": "success",
  "elapsed_seconds": 2847.3,
  "timestamp": "2026-04-12T10:30:00+00:00"
}
```

**Method-specific fields:**

**NUTS** additionally includes:
```json
{
  "num_warmup": 2000,
  "num_samples": 5000,
  "num_chains": 4,
  "target_accept_prob": 0.85,
  "max_tree_depth": 10,
  "parameterization": "ncp",
  "n_divergences": 0,
  "max_rhat": 1.003,
  "min_bulk_ess": 1847,
  "min_tail_ess": 1203,
  "converged": true
}
```

- `converged` = True iff `n_divergences == 0` AND `max_rhat < 1.01` AND `min_bulk_ess > 400`.

**ADVI** additionally includes:
```json
{
  "guide_type": "mean_field",
  "num_steps": 50000,
  "learning_rate": 0.01,
  "num_restarts": 5,
  "best_seed_index": 2,
  "final_elbo": -3847.2,
  "all_final_losses": [-3847.2, -3901.5, -3855.1, -3889.7, -3862.3],
  "loss_spread_std": 22.4,
  "num_posterior_samples": 5000,
  "lr_retries": 0
}
```

**Glasso** additionally includes:
```json
{
  "alpha_selected": 0.073,
  "cv_folds": 5,
  "n_nonzero_offdiag": 214
}
```

**Ledoit–Wolf** additionally includes:
```json
{
  "shrinkage_intensity": 0.42
}
```

**Sample Covariance** additionally includes:
```json
{
  "rank": 50,
  "invertible": true
}
```

### 3.3 The `metrics.json` Schema

Computed by the evaluation module against the ground truth Ω₀. Every method gets the same metrics:

```json
{
  "method": "nuts",
  "config_id": 7,
  "seed": 3,
  "steins_loss": 4.21,
  "frobenius_loss": 15.73,
  "frobenius_loss_relative": 0.127,
  "spectral_loss": 1.82,
  "tpr": 0.87,
  "fpr": 0.03,
  "precision": 0.91,
  "recall": 0.87,
  "f1": 0.89,
  "mcc": 0.84,
  "n_edges_detected": 112,
  "n_edges_true": 122,
  "eigenvalue_mse": 0.034,
  "condition_number_hat": 48.2,
  "condition_number_true": 42.1,
  "trace_error": 0.73,
  "gmv_weight_norm": 0.142,
  "oracle_gmv_weight_norm": 0.138,
  "edge_detection_method": "threshold_1e-5"
}
```

**For Bayesian methods, add:**
```json
{
  "edge_detection_method": "credible_interval_95",
  "coverage_95": 0.943,
  "mean_interval_width": 0.087,
  "mean_posterior_std_offdiag": 0.041,
  "bimodality_coefficient_kappa": 0.72,
  "shrinkage_wasserstein_vs_nuts": null
}
```

- `coverage_95`: fraction of true ω₀_ij values that fall within the 95% posterior credible interval. Should be ~0.95 for well-calibrated NUTS; expected to be lower for ADVI (overconfident).
- `bimodality_coefficient_kappa`: computed from the shrinkage coefficients κ̂_ij = E[1/(1 + λ²ᵢⱼτ²) | Y]. A bimodality coefficient > 5/9 ≈ 0.556 suggests the distribution is bimodal. NUTS should produce bimodal κ̂; ADVI-MF is expected to produce unimodal κ̂. **This is the paper's central diagnostic.**
- `shrinkage_wasserstein_vs_nuts`: Wasserstein-1 distance between the empirical distribution of κ̂ᵢⱼ from this method vs. NUTS. Set to `null` for NUTS itself. Computed only after both NUTS and ADVI have run for the same (config, seed).

### 3.4 Edge Detection for Sparsity Metrics

Different methods require different edge-detection strategies:

| Method | Edge Detection | Rationale |
|--------|---------------|-----------|
| `glasso` | `|ω̂_ij| > 0` (exact zeros) | Glasso produces exact structural zeros by construction. |
| `ledoit_wolf` | `|ω̂_ij| > threshold` | No structural zeros; use threshold = 1e-5. |
| `sample_cov` | `|ω̂_ij| > threshold` | Same as LW. |
| `nuts` | 95% credible interval excludes 0 | Bayesian edge detection: edge present iff P(ω_ij ≠ 0 | Y) > 0.95. |
| `advi_mf` | 95% credible interval excludes 0 | Same criterion, but from VI posterior. |
| `advi_fr` | 95% credible interval excludes 0 | Same. |

For the Bayesian methods, the credible interval criterion works as follows: for each off-diagonal (i,j) with i < j, compute the 2.5th and 97.5th percentiles of the posterior samples of ω_ij. If the interval [q_2.5, q_97.5] does not contain 0, declare the edge present.

**Also compute** a threshold-based version (`|posterior_mean_ij| > 1e-5`) for all Bayesian methods, to enable a fair apples-to-apples comparison with the frequentist threshold-based metrics. Store both in the metrics.

---

## 4. Computing Shrinkage Coefficients (Critical for the Paper)

The shrinkage coefficient is the diagnostic that tests the paper's central hypothesis.

### 4.1 Definition

For each off-diagonal pair (i,j) with i < j, the shrinkage coefficient at posterior sample s is:

```
κ_ij^(s) = 1 / (1 + λ_ij^(s)² · τ^(s)²)
```

where λ_ij^(s) and τ^(s) are the posterior samples of the local and global shrinkage parameters.

### 4.2 What to Compute and Save

1. **Per-sample κ matrix**: `kappa_samples.npy` of shape `(n_samples, n_offdiag)`. This is the raw material for all downstream shrinkage analysis.
2. **Posterior mean κ̂_ij**: `kappa_hat = kappa_samples.mean(axis=0)`. Shape `(n_offdiag,)`.
3. **Bimodality coefficient** of the empirical distribution of κ̂_ij across all (i,j) pairs:
   ```
   b = (skewness² + 1) / (kurtosis + 3(n-1)² / ((n-2)(n-3)))
   ```
   where `n = n_offdiag`, skewness and kurtosis are computed from `{κ̂_ij : i < j}`. If b > 5/9, the distribution is likely bimodal.

### 4.3 Why This Matters

The horseshoe's defining property is that κ_ij should be **bimodal**: values cluster near 0 (no shrinkage → signal → edge present) or near 1 (full shrinkage → noise → no edge). NUTS should preserve this bimodality. Mean-field ADVI is expected to **destroy** it by forcing a unimodal Gaussian approximation in the unconstrained space, leading to moderate shrinkage on everything (signals over-shrunk, noise under-shrunk). This is the paper's main hypothesis.

### 4.4 For Frequentist Methods

Frequentist methods don't have λ or τ, so they don't produce κ samples. However, you can compute an **analogous shrinkage profile**: the empirical distribution of |ω̂_ij| across all off-diagonal entries. For glasso, this distribution should be bimodal (exact zeros + nonzero estimates). For Ledoit–Wolf, it will be unimodal (everything shrunk toward the same target). Save this as `offdiag_magnitudes.npy` for plotting.

---

## 5. Implementation Architecture

### 5.1 Core Functions

```
src/inference/run_single.py:
  - run_inference(method, data_dir, output_dir, **method_kwargs) → diagnostics_dict
    # Dispatcher: loads Y from data_dir, runs the specified method, saves results to output_dir
    # Returns the diagnostics dict (also saved as diagnostics.json)

src/evaluation/evaluate_single.py:
  - evaluate(method, data_dir, results_dir) → metrics_dict
    # Loads omega_true from data_dir, omega_hat (+ samples) from results_dir
    # Computes all metrics, saves metrics.json, returns the dict

src/evaluation/metrics.py:  [extend existing]
  - steins_loss(Omega_hat, Omega_true) → float
  - frobenius_loss(Omega_hat, Omega_true) → float
  - spectral_loss(Omega_hat, Omega_true) → float
  - sparsity_metrics(Omega_hat, Omega_true, threshold) → dict
  - sparsity_metrics_credible(omega_samples, threshold_alpha=0.05) → edge_set
  - coverage_95(omega_samples, Omega_true) → float
  - bimodality_coefficient(kappa_hat_vector) → float
  - shrinkage_wasserstein(kappa_nuts, kappa_other) → float
  - eigenvalue_metrics(Omega_hat, Omega_true) → dict

src/evaluation/shrinkage.py:  [new]
  - compute_kappa_samples(tau_samples, lambda_samples) → kappa_samples
  - compute_kappa_hat(kappa_samples) → kappa_hat
  - shrinkage_profile_summary(kappa_hat) → dict (bimodality coeff, quartiles, etc.)

scripts/run_inference_single.py:
  - CLI: --config-id N --seed S --method M --data-root ... --results-root ...
  # Thin wrapper: resolves the data_dir and output_dir from the manifest, calls run_inference + evaluate

scripts/run_inference_slurm.sh:
  - SLURM array job script (see §7)

scripts/audit_results.py:
  - Post-run audit: walk results/, check for missing/corrupted outputs
```

### 5.2 The `run_inference` Dispatcher (Pseudocode)

```python
def run_inference(method, data_dir, output_dir, **kwargs):
    # 1. Load data
    Y = np.load(data_dir / "Y.npy")
    metadata = json.load(open(data_dir / "metadata.json"))
    p, T = metadata["p"], metadata["T"]

    # 2. Dispatch to method
    start = time.time()
    if method == "nuts":
        result = _run_nuts(Y, p, **kwargs)
    elif method == "advi_mf":
        result = _run_advi(Y, p, guide_type="mean_field", **kwargs)
    elif method == "advi_fr":
        result = _run_advi(Y, p, guide_type="full_rank", **kwargs)
    elif method == "glasso":
        result = _run_glasso(Y, **kwargs)
    elif method == "ledoit_wolf":
        result = _run_ledoit_wolf(Y, **kwargs)
    elif method == "sample_cov":
        result = _run_sample_cov(Y, p, T)
    elapsed = time.time() - start

    # 3. Build diagnostics
    diagnostics = {
        "method": method,
        "config_id": metadata["config_id"],
        "seed": metadata["seed"],
        "p": p, "T": T,
        "status": result["status"],
        "elapsed_seconds": elapsed,
        **result.get("diagnostics", {})
    }

    # 4. Save outputs atomically (same pattern as WORK1)
    _save_results(output_dir, result, diagnostics)

    return diagnostics
```

### 5.3 Extracting Posterior Quantities from NUTS

After `mcmc.run(...)`:

```python
samples = mcmc.get_samples()

# Off-diagonal Ω entries
if "omega_offdiag" in samples:
    omega_offdiag = samples["omega_offdiag"]  # (n_samples, n_offdiag)
else:
    # Non-centered: reconstruct
    z = samples["z"]
    lambdas = samples["lambdas"]
    tau = samples["tau"]
    omega_offdiag = z * lambdas * tau[:, None]  # broadcast

# Diagonal
omega_diag = samples["omega_diag"]  # (n_samples, p)

# Full matrices via vmap
from jax import vmap
omega_samples = vmap(lambda od, d: assemble_precision_matrix(od, d, p))(
    omega_offdiag, omega_diag
)  # (n_samples, p, p)

# Shrinkage coefficients
tau_samples = samples["tau"]
lambda_samples = samples["lambdas"]
kappa_samples = 1.0 / (1.0 + lambda_samples**2 * tau_samples[:, None]**2)

# Point estimate
omega_hat = omega_samples.mean(axis=0)

# Diagnostics from mcmc.print_summary()
summary = mcmc.get_extra_fields()
# Or use numpyro.diagnostics directly
from numpyro.diagnostics import summary as numpyro_summary
```

### 5.4 Extracting Posterior Quantities from ADVI

After `svi.run(...)`:

```python
from numpyro.infer import Predictive

predictive = Predictive(guide, params=svi_result.params, num_samples=5000)
vi_samples = predictive(rng_key, Y=Y, p=p)

# Same extraction logic as NUTS (the sample sites have the same names)
# Then same kappa computation
```

### 5.5 Handling Failures Gracefully

Every method wrapper must catch exceptions and return a result dict with `status: "failed"` or `status: "timeout"` rather than crashing. This is critical for cluster runs where one failure shouldn't abort the entire array.

```python
def _run_nuts(Y, p, **kwargs):
    try:
        # ... run NUTS ...
        if elapsed > kwargs.get("timeout", 14400):  # 4 hours
            return {"status": "timeout", "omega_hat": None}
        if n_divergences > 0.1 * num_samples:
            # Retry with centered parameterization
            ...
        return {"status": "success", "omega_hat": omega_hat, ...}
    except Exception as e:
        return {"status": "failed", "error": str(e), "omega_hat": None}
```

When `status` is not `"success"`, the evaluation step should still write a `metrics.json` with all metric values set to `null`, so that downstream aggregation scripts can distinguish "method failed" from "method not yet run".

---

## 6. Evaluation Details

### 6.1 Stein's Loss

```
L_S(Ω̂, Ω₀) = tr(Ω̂⁻¹ Ω₀) − log|Ω̂⁻¹ Ω₀| − p
```

Implementation: compute M = solve(Ω̂, Ω₀) via `numpy.linalg.solve` (avoids explicit inversion), then `L_S = trace(M) - slogdet(M)[1] - p`.

**Edge case**: if Ω̂ is singular or nearly singular (sample covariance at high γ), `numpy.linalg.solve` will fail. Catch this and return `steins_loss: null`.

### 6.2 Frobenius Loss

```
L_F = ||Ω̂ − Ω₀||_F² = Σ_{i,j} (ω̂_ij − ω₀_ij)²
```

Also compute the relative version: `L_F_rel = L_F / ||Ω₀||_F²`.

### 6.3 Spectral Loss

```
L_2 = ||Ω̂ − Ω₀||_2 = σ_max(Ω̂ − Ω₀)
```

### 6.4 Sparsity Recovery

For threshold-based detection:
```python
pred_edges = set((i, j) for i, j in zip(*np.where(np.abs(Omega_hat) > threshold)) if i < j)
true_edges = set(tuple(e) for e in metadata["edge_set"])
```

Then compute TP, FP, FN, TN and derive TPR, FPR, precision, recall, F1, MCC.

For credible-interval-based detection (Bayesian methods):
```python
q_lo = np.percentile(omega_samples[:, i, j], 2.5)
q_hi = np.percentile(omega_samples[:, i, j], 97.5)
edge_present = not (q_lo <= 0 <= q_hi)
```

### 6.5 Posterior Calibration (Bayesian Only)

For each off-diagonal entry (i,j):
1. Compute the 95% credible interval [q_2.5, q_97.5] from posterior samples.
2. Check if the true value ω₀_ij falls within this interval.
3. `coverage_95` = fraction of entries where the true value is covered.

Well-calibrated inference → coverage ≈ 0.95. ADVI is expected to have coverage < 0.95 (too narrow intervals = overconfident).

### 6.6 Eigenvalue Metrics

Compare sorted eigenvalues of Ω̂ against sorted eigenvalues of Ω₀:
```
eigenvalue_mse = (1/p) Σ_i (λ̂_i − λ₀_i)²
```

Also store the condition number of Ω̂ for comparison against κ(Ω₀).

---

## 7. SLURM Orchestration

### 7.1 The Fundamental Challenge: Heterogeneous Runtimes

Unlike WORK1 where every task took < 15 seconds, inference runtimes vary by 4+ orders of magnitude:

| Method | p=10 | p=50 | p=100 |
|--------|------|------|-------|
| `sample_cov` | <1s | <1s | <1s |
| `ledoit_wolf` | <1s | <1s | <1s |
| `glasso` | <5s | <30s | <2min |
| `advi_mf` | <1min | 5–15min | 30–60min |
| `advi_fr` | <2min | 15–60min | infeasible / LR fallback |
| `nuts` | <2min | 30–120min | 1–4h (may timeout) |

### 7.2 Strategy: Three Separate Job Arrays

Rather than one monolithic array, submit **three SLURM arrays** with different resource profiles:

#### Array 1: Frequentist Methods (fast)

- **Methods**: `sample_cov`, `ledoit_wolf`, `glasso`
- **Parallelism**: batch all 3 methods × all 20 seeds for one config into a single task.
- **Array size**: 84 tasks (one per config).
- **Resources**: 1 CPU, 4 GB, 30 min wall time.
- Each task iterates: `for method in [sample_cov, ledoit_wolf, glasso]: for seed in range(20): run + evaluate`.

#### Array 2: ADVI Methods (medium)

- **Methods**: `advi_mf`, `advi_fr`
- **Parallelism**: one task per (config_id, seed, method) triple.
- **Array size**: 84 configs × 20 seeds × 2 methods = 3,360 tasks. But SLURM array max index is typically 10000, so this fits. If the cluster limits concurrent jobs, use `%50` to cap concurrency.
- **Resources**: 1 CPU (ADVI is not easily multi-core), 8 GB, 2 hours wall time.
- For p=100 advi_fr: the task will auto-detect infeasibility and fall back to low-rank.

#### Array 3: NUTS (heavy)

- **Methods**: `nuts` only
- **Parallelism**: one task per (config_id, seed).
- **Array size**: 84 × 20 = 1,680 tasks.
- **Resources**: 1 CPU (or 1 GPU if available — NumPyro/JAX on GPU gives ~3–5× speedup for NUTS), 16 GB, 4 hours wall time.
- For p=100: many tasks will timeout. This is expected. The diagnostics.json will record `status: "timeout"`.

### 7.3 Task ID Encoding

Each task in arrays 2 and 3 needs to decode a SLURM array index into (config_id, seed, [method]).

**Convention**: `task_id = config_id * n_seeds * n_methods + seed * n_methods + method_index`

Or more simply, pre-generate a **task manifest** — a CSV/JSON file mapping `task_id → (config_id, seed, method)` — and have each task look up its assignment. This is more robust and debuggable than arithmetic encoding.

```python
# scripts/generate_task_manifest.py
tasks = []
for config in manifest:
    for seed in range(n_seeds):
        for method in methods:
            tasks.append({"task_id": len(tasks), "config_id": config["config_id"],
                          "seed": seed, "method": method})
json.dump(tasks, open("task_manifest_advi.json", "w"))
```

### 7.4 Example SLURM Script (NUTS)

```bash
#!/bin/bash
#SBATCH --job-name=nuts_infer
#SBATCH --array=0-1679%100
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nuts_%A_%a.out
#SBATCH --error=logs/nuts_%A_%a.err

module load miniforge
conda activate ggm_horseshoe

python scripts/run_inference_single.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --task-manifest results/task_manifest_nuts.json \
    --data-root data/synthetic/ \
    --results-root results/synthetic/ \
    --config-manifest data/synthetic/configs/config_manifest.json
```

The `%100` after the array range limits SLURM to running at most 100 tasks concurrently, to be a good citizen on the shared cluster.

---

## 8. Priority Ordering (Progress Report by April 15)

### Tier 0: Smoke Test (do first, takes 5 minutes)

Run NUTS on **one seed** at **p=10, T=100** (config with γ=0.10, s=0.10, ER — this is the easiest setting). Verify:
- R̂ < 1.01 for all parameters
- 0 divergent transitions
- Posterior mean Ω̂ is visually close to Ω₀ (plot heatmaps)
- Stein's loss is small (< 5)

**If this fails, stop and debug the model before running anything else.**

### Tier 1: Progress Report Slice (do second, submit to cluster)

Run **all 6 methods** on:
- p=50, γ ∈ {0.67, 0.20} (i.e., T=75 and T=250) — one hard, one moderate
- graph = erdos_renyi
- s = 0.10
- **seeds 0–4** (5 seeds — enough for error bars)

This is 2 configs × 5 seeds × 6 methods = **60 inference runs**. The NUTS runs (~10 at p=50) will take ~30–120 min each; everything else finishes in minutes.

**Deliverables from Tier 1:**
- One precision matrix heatmap comparison (6 panels: Ω₀, sample cov, LW, glasso, NUTS, ADVI-MF) for p=50, T=250, seed 0.
- One shrinkage profile plot (NUTS κ̂ vs ADVI-MF κ̂ histograms) — **the paper's key figure**.
- One metrics comparison table (Stein's loss, F1, MCC across all 6 methods, with std across seeds).

### Tier 2: Core Results (after progress report)

Extend to:
- p=50, all 5 γ values, ER, s=0.10, 5 seeds → 25 configs × 5 seeds × 6 methods = 750 runs
- Add block-diagonal → 50 × 5 × 6 = 1,500 runs
- Add s ∈ {0.05, 0.30} → full p=50 slice

This produces the **loss-vs-γ curves** (the paper's second key figure).

### Tier 3: Full Grid (if compute allows)

All 84 configs × 20 seeds × 6 methods. This is ~10,000 runs. NUTS at p=100 will timeout on many configs — document this as a finding.

---

## 9. Aggregation and Summary Tables

After inference + evaluation complete, build aggregate summaries.

### 9.1 Per-Config Summary

For each (config_id, method), aggregate across seeds:

```json
{
  "config_id": 7,
  "method": "nuts",
  "n_seeds_success": 5,
  "n_seeds_failed": 0,
  "steins_loss_mean": 4.21,
  "steins_loss_std": 0.83,
  "frobenius_loss_mean": 15.73,
  "f1_mean": 0.89,
  "mcc_mean": 0.84,
  "coverage_95_mean": 0.943,
  "bimodality_coeff_mean": 0.72,
  "elapsed_mean": 2847.3
}
```

### 9.2 Cross-Method Comparison Table

For each config, produce a table like:

| Method | Stein's Loss | Frobenius | F1 | MCC | Coverage | Bimodality | Time (s) |
|--------|-------------|-----------|-----|-----|----------|------------|----------|
| NUTS | 4.2 ± 0.8 | 15.7 ± 2.1 | 0.89 ± 0.03 | 0.84 ± 0.04 | 0.94 | 0.72 | 2847 |
| ADVI-MF | 5.1 ± 1.2 | 18.3 ± 3.0 | 0.81 ± 0.05 | 0.76 ± 0.06 | 0.82 | 0.38 | 423 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### 9.3 Scripts for Aggregation

```
scripts/aggregate_results.py:
  - Walks results/synthetic/, reads all metrics.json files, produces:
    - results/summary/per_config_method.json (one row per config × method)
    - results/summary/cross_method_table.json (pivot table)
    - results/summary/loss_vs_gamma.json (for plotting)

scripts/generate_figures.py:
  - Reads summary files, produces publication-quality figures:
    - figures/heatmap_comparison.pdf
    - figures/shrinkage_profiles.pdf
    - figures/loss_vs_gamma.pdf
    - figures/roc_curves.pdf
    - figures/runtime_comparison.pdf
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```
tests/test_inference.py:
  - test_nuts_smoke_p5: Run NUTS on p=5, T=500 synthetic data. Verify convergence.
  - test_advi_mf_smoke_p5: Run ADVI-MF on same data. Verify ELBO converges.
  - test_advi_fr_smoke_p5: Run ADVI-FR on same data.
  - test_glasso_returns_sparse: Verify glasso Ω̂ has exact zeros.
  - test_ledoit_wolf_returns_pd: Verify LW Ω̂ is PD.
  - test_sample_cov_singular_when_p_gt_T: Verify returns None when T < p.

tests/test_metrics.py:
  - test_steins_loss_zero_at_truth: L_S(Ω₀, Ω₀) == 0.
  - test_steins_loss_positive: L_S(Ω̂, Ω₀) > 0 for Ω̂ ≠ Ω₀.
  - test_frobenius_loss_zero_at_truth: same.
  - test_sparsity_perfect_at_truth: TPR=1, FPR=0 when Ω̂ = Ω₀.
  - test_coverage_perfect_at_truth: coverage=1.0 when samples are delta at Ω₀.
  - test_bimodality_bimodal: synthetic bimodal data → coefficient > 5/9.
  - test_bimodality_unimodal: synthetic unimodal data → coefficient < 5/9.
  - test_kappa_computation: verify κ = 1/(1 + λ²τ²) on known values.

tests/test_evaluation_integration.py:
  - test_full_pipeline_p10: Generate data, run all 6 methods, evaluate, verify all metrics.json files exist and are parseable with reasonable values.
```

### 10.2 Integration Smoke Test (Local, Before Cluster)

Run the full pipeline on **one seed** at **p=10** for all 6 methods. Verify:
- All 6 `diagnostics.json` files exist and have `status: "success"`.
- All 6 `metrics.json` files exist with non-null values (except sample_cov which may be null at high γ).
- NUTS R̂ < 1.01, 0 divergences.
- ADVI ELBO converged (final loss < initial loss).
- Stein's loss ranking makes intuitive sense (oracle < NUTS ≤ glasso < LW < sample_cov, roughly).

---

## 11. What NOT to Do

- **Do not modify the data in `data/synthetic/`.** It is read-only input. All outputs go to `results/synthetic/`.
- **Do not run NUTS with fewer than 4 chains** (except for the p=5 unit test). R̂ requires multiple chains.
- **Do not skip the ADVI multi-seed restarts.** The ELBO landscape is multimodal for the horseshoe; a single seed may find a bad local optimum and you wouldn't know.
- **Do not use ADVI posterior samples to compute R̂.** R̂ is an MCMC diagnostic; it has no meaning for VI. Use ELBO convergence and PSIS-k̂ instead.
- **Do not threshold the horseshoe posterior mean at 1e-5 and call it "edge detection."** The Bayesian way is the credible interval criterion. Compute both, but the credible interval version is the primary one for the paper.
- **Do not compare ADVI κ̂ to NUTS κ̂ unless they ran on the same (config_id, seed) pair.** Cross-seed comparison is meaningless because the ground truth differs.
- **Do not hard-code paths.** Everything derives from the config manifest + a base data root + a base results root.
- **Do not re-generate synthetic data.** Load it from WORK1's output.
- **Do not run the full grid before the smoke test passes.** A bug in the model or the inference wrapper will waste hundreds of GPU-hours.

---

## 12. Deliverables Checklist

When WORK2 is complete, the following should exist and be verified:

### Code

- [ ] `src/inference/run_single.py` — dispatcher that runs one method on one seed
- [ ] `src/evaluation/evaluate_single.py` — computes all metrics for one (method, seed) pair
- [ ] `src/evaluation/metrics.py` — completed with all loss functions + sparsity metrics + calibration
- [ ] `src/evaluation/shrinkage.py` — κ computation and bimodality diagnostics
- [ ] `scripts/run_inference_single.py` — CLI entry point for one (config, seed, method)
- [ ] `scripts/generate_task_manifests.py` — produces task manifests for each SLURM array
- [ ] `scripts/run_freq_slurm.sh` — SLURM array for frequentist methods
- [ ] `scripts/run_advi_slurm.sh` — SLURM array for ADVI methods
- [ ] `scripts/run_nuts_slurm.sh` — SLURM array for NUTS
- [ ] `scripts/audit_results.py` — post-run audit
- [ ] `scripts/aggregate_results.py` — aggregation across seeds
- [ ] `tests/test_inference.py` — unit tests for inference wrappers
- [ ] `tests/test_metrics.py` — unit tests for evaluation functions
- [ ] `tests/test_evaluation_integration.py` — integration smoke test

### Data (on cluster)

- [ ] Smoke test at p=10 passes: NUTS converges, all 6 methods produce valid outputs
- [ ] Tier 1 results (p=50, 2 γ values, 5 seeds, 6 methods) complete with 0 failures
- [ ] All `diagnostics.json` and `metrics.json` files are parseable
- [ ] Aggregated summary tables exist at `results/summary/`

### For the Progress Report (April 15)

- [ ] Precision matrix heatmap comparison figure (6 panels)
- [ ] Shrinkage profile comparison figure (NUTS κ̂ vs ADVI-MF κ̂)
- [ ] Metrics comparison table (Stein's loss, F1, MCC, runtime)
- [ ] One-paragraph description of the "loss vs p/T" plot that will appear in the final report (required by the progress report spec: "describe the general layout of at least one plot")