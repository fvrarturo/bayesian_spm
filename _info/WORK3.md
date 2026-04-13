# Agent Prompt: WORK3 — Grid Expansion, Gibbs Sampler, Real Data, Final Report & Presentation (REVISED)

## Context

You are completing the **final phase** of a research project comparing MCMC (NUTS) vs. Variational Inference (ADVI) for sparse Bayesian precision matrix estimation using the graphical horseshoe prior. MIT course 6.7830, Spring 2026.

**Timeline:**
- **April 15** (3 days): Progress report due (6 pages max)
- **May 5** (23 days): Final report due (12 pages max, ICML style)
- **Final class meetings** (early May): Presentation (10–15 min)

**What's done (WORK1 + WORK2):**
- 1,680 synthetic datasets (84 configs × 20 seeds), fully validated.
- Complete inference + evaluation pipeline. Tier 1 results (p=50, ER, s=0.10, γ ∈ {0.67, 0.20}, 5 seeds).
- 5 of 6 methods succeeded: NUTS, ADVI-MF, glasso, Ledoit-Wolf, sample covariance. ADVI-FR timed out.
- 102 tests pass.

**Key findings from Tier 1:**
1. NUTS dominates on Stein's loss (2–5×) and calibration (coverage 0.96–0.99).
2. ADVI-MF: 2× worse Stein's loss, coverage 3–5 points below nominal.
3. **Surprise**: ADVI-MF preserves bimodal shrinkage (b ≈ 0.88–0.90). Original hypothesis of bimodality collapse NOT confirmed.
4. ADVI-MF has slightly better F1 at the hard setting (overconfident intervals are more aggressive).
5. Frequentist methods clearly dominated. Glasso has extremely high Stein's loss variance.

**What this prompt adds beyond the original WORK3:**
- **Li et al. (2019) Gibbs sampler** as a new MCMC method (ties to Lecture 8: Gibbs sampling)
- **Full expansion to p ∈ {10, 50, 100}** promoted to core Tier 2 (not optional Tier 3)
- **PSIS diagnostics** for evaluating VI approximation quality (ties to Yao et al. 2018)
- **Sparsity sensitivity (s = 0.05, 0.10, 0.30)**: s = 0.10 remains the main case for all core results, but if time allows, running s = 0.05 (very sparse) is especially illuminating. The s = 0.05 case is where the horseshoe prior's ability to aggressively shrink to zero is hypothesized to show the strongest contrast to glasso and other competitors.

---

## 1. The Seven Methods

### 1.1 Updated Method Table

| Method ID | Full Name | Type | Implementation | Course Lecture |
|-----------|-----------|------|---------------|----------------|
| `nuts` | NUTS | Gradient MCMC | NumPyro | Lecture 10: NUTS |
| `gibbs` | Li et al. Gibbs | Model-specific MCMC | Custom Python/NumPy | Lecture 8: MCMC/Gibbs |
| `advi_mf` | ADVI Mean-Field | Variational | NumPyro SVI + AutoNormal | Lecture 5: ADVI |
| `advi_lr` | ADVI Low-Rank | Variational | NumPyro SVI + AutoLowRankMVN | Lecture 3: SVI |
| `glasso` | Graphical Lasso (CV) | Frequentist | sklearn | — |
| `ledoit_wolf` | Ledoit–Wolf | Frequentist | sklearn | — |
| `sample_cov` | Sample Covariance | Frequentist | numpy | — |

### 1.2 Why These Seven

The methods span the full inference spectrum covered in the course:

- **NUTS** (Lectures 9–10): Generic gradient-based MCMC. Automatic trajectory tuning. The gold standard for posterior quality but expensive.
- **Gibbs** (Lecture 8): Model-specific coordinate-wise MCMC. Hand-derived conditionals, 100% acceptance rate, no gradients needed. The method from the original Li et al. paper. Tests whether a bespoke sampler outperforms a generic one.
- **ADVI-MF** (Lecture 5): Automated mean-field VI. Fast but loses posterior correlations.
- **ADVI-LR** (Lecture 3/SVI): Low-rank covariance VI. Richer than MF, cheaper than full-rank. Tests whether capturing some posterior correlation improves calibration.
- **Glasso / LW / Sample Cov**: Frequentist baselines showing what you get without Bayesian regularization.

The **three-way MCMC comparison** (NUTS vs. Gibbs vs. frequentist) directly tests the tradeoff discussed in Lecture 8: Gibbs has 100% acceptance and no gradient cost, but component-wise updates mix slowly in correlated posteriors. NUTS makes joint proposals using gradient geometry, potentially mixing faster but at higher per-step cost.

---

## 2. The Li et al. Gibbs Sampler — Full Specification

### 2.1 Algorithm Overview

The Gibbs sampler from Li, Craig, and Bhadra (2019) exploits the **column-wise structure** of the precision matrix. One full sweep updates all p columns of Ω, all local shrinkage parameters λ²_ij, and the global shrinkage τ².

**Algorithm: One Gibbs Sweep**

```
Input: Current state (Ω, {λ²_ij}, {ν_ij}, τ², ξ), scatter matrix S = YᵀY, sample size T

# --- Block 1: Update Ω column by column ---
For j = 1, ..., p:
    # Partition Ω and S around column j
    Ω_{-j,-j} = Ω with row j and col j removed           # (p-1) × (p-1)
    ω_{-j,j}  = column j of Ω, excluding diagonal         # (p-1) × 1
    s_{-j,j}  = column j of S, excluding diagonal          # (p-1) × 1
    s_jj       = S[j,j]                                     # scalar

    # Construct the conditional precision for column j
    D_j = diag(1 / (λ²_{1j} · τ²), ..., 1 / (λ²_{(p-1)j} · τ²))   # prior precision
    C_j = inv(s_jj · Ω_{-j,-j} + D_j)                               # (p-1) × (p-1)

    # Sample off-diagonal entries of column j
    μ_j = -C_j @ s_{-j,j} / s_jj
    Σ_j = C_j / (s_jj + T)                                # NOTE: see §2.3 for the exact formula
    ω_{-j,j} ~ N(μ_j, Σ_j)                                # truncated to keep Ω ≻ 0

    # Sample diagonal entry
    ω_jj ~ Gamma(T/2 + 1, s_jj / 2) + ω_{-j,j}ᵀ @ inv(Ω_{-j,-j}) @ ω_{-j,j}

# --- Block 2: Update local shrinkage λ²_ij ---
For all i < j:
    ν_ij  ~ InvGamma(1, 1 + 1/λ²_ij)
    λ²_ij ~ InvGamma(1, 1/ν_ij + ω²_ij / (2τ²))

# --- Block 3: Update global shrinkage τ² ---
ξ  ~ InvGamma(1, 1 + 1/τ²)
τ² ~ InvGamma((p(p-1)/2 + 1) / 2,  1/ξ + (1/2) Σ_{i<j} ω²_ij / λ²_ij)
```

### 2.2 Implementation Details

**File**: `src/inference/gibbs_runner.py`

**Key implementation decisions:**

#### 2.2.1 The Truncated Normal Sampling

The off-diagonal update `ω_{-j,j} ~ N(μ_j, Σ_j)` is subject to the constraint that the full Ω remains positive definite. In Li et al., this is handled by **rejection**: sample from the untruncated normal, check if Ω ≻ 0, and resample if not.

In practice, for moderate p and reasonable starting points, the rejection rate is very low (< 1%) because the prior already concentrates the off-diagonals near zero. Implement as:

```python
def sample_column_offdiag(mu_j, Sigma_j, Omega, j, max_attempts=100):
    """Sample ω_{-j,j} from N(mu_j, Sigma_j) truncated to Ω ≻ 0."""
    for attempt in range(max_attempts):
        proposal = rng.multivariate_normal(mu_j, Sigma_j)
        # Insert proposal into Ω and check PD
        Omega_candidate = Omega.copy()
        idx = [i for i in range(p) if i != j]
        Omega_candidate[idx, j] = proposal
        Omega_candidate[j, idx] = proposal
        if np.linalg.eigvalsh(Omega_candidate).min() > 0:
            return proposal
    # If rejection fails, return the mean (safe fallback)
    warnings.warn(f"Truncated normal rejection failed after {max_attempts} attempts for column {j}")
    return mu_j
```

**Optimization**: instead of a full eigendecomposition (O(p³)) per rejection check, use the **Schur complement**: Ω ≻ 0 iff ω_jj − ω_{-j,j}ᵀ Ω⁻¹_{-j,-j} ω_{-j,j} > 0. Since Ω_{-j,-j} is unchanged, precompute its inverse once per column. This reduces the check to O(p²).

#### 2.2.2 The Shifted Gamma for the Diagonal

The conditional for ω_jj is:

```
ω_jj | rest ~ Gamma(T/2 + 1, s_jj/2) + ω_{-j,j}ᵀ Ω⁻¹_{-j,-j} ω_{-j,j}
```

This is a Gamma distribution **shifted** by the quadratic form. Sample `g ~ Gamma(T/2 + 1, scale = 2/s_jj)` and then set `ω_jj = g + ω_{-j,j}ᵀ @ inv(Ω_{-j,-j}) @ ω_{-j,j}`.

Note: NumPy's `numpy.random.Generator.gamma(shape, scale)` uses the shape/scale parameterization. The shape is `T/2 + 1`, the scale is `2/s_jj`.

#### 2.2.3 The Data-Augmentation Trick for Half-Cauchy

The half-Cauchy prior on λ_ij is not conjugate, so Li et al. introduce auxiliary variables ν_ij to create a conditionally conjugate hierarchy:

```
λ²_ij | ν_ij ~ InvGamma(1/2, 1/ν_ij)
ν_ij         ~ InvGamma(1/2, 1)
```

Marginally, `λ_ij ~ C⁺(0, 1)`. The posterior conditionals become:

```
ν_ij | λ²_ij      ~ InvGamma(1, 1 + 1/λ²_ij)
λ²_ij | ω_ij, τ²  ~ InvGamma(1, 1/ν_ij + ω²_ij / (2τ²))
```

Same pattern for τ² with auxiliary ξ:

```
ξ | τ²  ~ InvGamma(1, 1 + 1/τ²)
τ² | ... ~ InvGamma((n_offdiag + 1)/2, 1/ξ + (1/2) Σ_{i<j} ω²_ij / λ²_ij)
```

where `n_offdiag = p(p-1)/2`.

Use `scipy.stats.invgamma` or sample as `1 / Gamma(shape, scale)`.

#### 2.2.4 Initialization

Initialize at the same safe starting point as NUTS:
- `Ω = diag(ω_diag)` where `ω_diag_i ~ Gamma(2, 0.5)` (so diagonal entries are ~4, giving a PD matrix)
- `ω_{ij} = 0` for all off-diagonal entries (trivially PD)
- `λ²_ij = 1` for all i < j
- `τ² = 1`
- `ν_ij = 1`, `ξ = 1`

#### 2.2.5 Run Configuration

| Parameter | p=10 | p=50 | p=100 |
|-----------|------|------|-------|
| Burn-in sweeps | 1,000 | 2,000 | 3,000 |
| Post-burn-in sweeps | 5,000 | 5,000 | 5,000 |
| Thinning | 1 | 1 | 2 |
| Total samples saved | 5,000 | 5,000 | 2,500 |
| Chains | 1 | 1 | 1 |

**Why 1 chain (not 4)?** The Gibbs sampler does not support parallel chains in the same way as NUTS. Running 4 sequential chains would quadruple wall time. Instead, run 1 long chain and assess convergence via:
- Trace plots of τ², a few representative ω_ij
- Geweke diagnostic (compare first 10% vs. last 50% of the chain)
- Effective sample size (ESS) via autocorrelation

**Expected wall time**: each sweep is O(p³) from the p matrix inversions. For p=50 with 7,000 total sweeps, expect 10–30 minutes (much faster than NUTS's 80–120 min). For p=100 with 8,000 sweeps, expect 1–3 hours (may succeed where NUTS times out).

### 2.3 What the Gibbs Sampler Saves

Same output format as NUTS:

```
results/synthetic/<graph>/<p>/<gamma>/<s>/seed_NN/gibbs/
    omega_hat.npy           # (p, p) — posterior mean
    omega_samples.npy       # (n_samples, p, p) — thinned samples, float32
    tau_samples.npy         # (n_samples,)
    lambda_samples.npy      # (n_samples, n_offdiag)
    kappa_samples.npy       # (n_samples, n_offdiag)
    diagnostics.json        # Gibbs-specific diagnostics
    metrics.json            # evaluation against ground truth
```

**Gibbs-specific diagnostics.json fields:**
```json
{
  "method": "gibbs",
  "n_burnin": 2000,
  "n_samples": 5000,
  "n_thinning": 1,
  "n_chains": 1,
  "n_rejection_failures": 0,
  "mean_rejection_rate_per_column": 0.003,
  "max_rejection_rate_per_column": 0.012,
  "min_ess_omega": 820,
  "min_ess_tau": 450,
  "geweke_p_value_tau": 0.73,
  "elapsed_seconds": 1247.3
}
```

### 2.4 The Course Connection

The Gibbs sampler is the **direct link between the project and Lecture 8 (MCMC)**. In the final report, frame the NUTS vs. Gibbs comparison as:

> "The Gibbs sampler of Li et al. (2019) exploits the column-wise conditional structure of the precision matrix, sampling each column from a truncated multivariate normal with 100% acceptance rate (cf. Lecture 8, where we showed Gibbs proposals are always accepted because they sample from the exact conditional). NUTS, by contrast, treats the posterior as a black box and uses gradient information to make large joint proposals. We compare these approaches to empirically test whether model-specific structure or generic gradient geometry provides more efficient exploration."

---

## 3. PSIS Diagnostics for VI Quality

### 3.1 What Is PSIS?

Pareto Smoothed Importance Sampling (PSIS) is a diagnostic for evaluating how well a variational approximation q(θ) matches the true posterior p(θ|y). It works by importance-weighting VI samples:

```
w_s = p(θ_s | y) / q(θ_s)  for θ_s ~ q
```

The distribution of log importance weights is fit to a generalized Pareto distribution. The shape parameter k̂ measures the mismatch:

- **k̂ < 0.5**: q is a good approximation. Importance sampling from q to p works well.
- **0.5 < k̂ < 0.7**: marginal. Results may be unreliable.
- **k̂ > 0.7**: q is a poor approximation. The tails of p are much heavier than q, and importance sampling fails.

### 3.2 Implementation

Use ArviZ's built-in PSIS:

```python
import arviz as az

# Compute log importance weights
# log w_s = log p(y, θ_s) - log q(θ_s)
# where θ_s are samples from the VI guide
log_weights = log_joint_vi_samples - log_q_vi_samples

# ArviZ PSIS
psis_result = az.psislw(log_weights)
# or:
# from arviz.stats import psislw
# lw_psis, khat = psislw(log_weights)
khat = psis_result.pareto_k  # scalar or per-observation
```

**Computing log p(y, θ_s)**: evaluate the NumPyro model's log-joint at each VI sample. NumPyro provides `log_density(model, model_args, model_kwargs, params)` for this.

**Computing log q(θ_s)**: evaluate the guide's log-density at each sample. For `AutoNormal`, this is a product of independent normals in unconstrained space (straightforward). For `AutoLowRankMVN`, it requires evaluating the low-rank Gaussian density.

### 3.3 When to Compute PSIS

Compute PSIS for every ADVI run (both MF and LR). Add to `diagnostics.json`:

```json
{
  "psis_khat": 0.62,
  "psis_interpretation": "marginal"
}
```

And in `metrics.json`:

```json
{
  "psis_khat": 0.62
}
```

### 3.4 The Course Connection

PSIS ties to:
- **Lecture 4** (Turner & Sahani): diagnosing when variational approximations are problematic.
- **Lecture 5** (ADVI): the question "when is mean-field a risky choice?"
- **Yao et al. (2018)** "Yes, but did it work?" from the reading plan (P3 priority).

In the report: "We use PSIS-k̂ (Vehtari et al., 2015; Yao et al., 2018) to diagnose the quality of the variational approximation without requiring the true posterior. Values k̂ > 0.7 indicate that the mean-field family is a poor match for the horseshoe posterior, consistent with the calibration degradation observed in the coverage analysis."

---

## 4. Full Expansion to p ∈ {10, 50, 100} and Sparsity Levels

### 4.1 Why All Three p Values Matter

| p | Role | NUTS feasible? | Gibbs feasible? | Narrative |
|---|------|---------------|-----------------|-----------|
| 10 | **Validation anchor** | Yes (< 2 min) | Yes (< 1 min) | Everything should work perfectly. If NUTS ≠ Gibbs ≠ truth, something is wrong. |
| 50 | **Main results** | Yes (80–120 min) | Yes (10–30 min) | Core comparison. All 7 methods run. |
| 100 | **Scalability frontier** | Likely timeout (4h+) | Likely feasible (1–3h) | NUTS infeasible → practitioner must choose Gibbs, ADVI, or frequentist. **Gibbs may be the only viable exact MCMC method at p=100.** |

The p=100 story is much stronger with the Gibbs sampler: instead of "NUTS fails at p=100 so you're stuck with ADVI," it becomes "NUTS fails but the model-specific Gibbs sampler still produces exact posterior samples, at the cost of requiring hand-derived conditionals."

### 4.2 The Expanded Grid

**Core grid (Tier 2):**

| Factor | Levels |
|--------|--------|
| p | 10, 50, 100 |
| γ | 0.90, 0.67, 0.42, 0.20, 0.10 |
| graph | erdos_renyi |
| s (sparsity) | 0.10 |
| seeds | 0–4 (5 seeds) |
| methods | all 7 |

This gives 15 (p, γ) pairs × 5 seeds × 7 methods = **525 inference tasks**.

**Sparsity note:** The main case throughout is s = 0.10, reflecting moderate sparsity. If time and compute allow, also run **s = 0.05** (very sparse) in at least one representative setting (e.g., p=50, γ=0.42). That's where the horseshoe's advantage should be most pronounced relative to glasso and other non-Bayesian baselines. Optionally, s = 0.30 (much less sparse) can also be included for sensitivity.

**Sensitivity extensions (Tier 2b):**

| Factor | Levels | Fixed |
|--------|--------|-------|
| s ∈ {0.05, 0.30} | p=50, γ=0.42 | ER graph |
| block-diagonal graph | p=50, γ ∈ {0.67, 0.42, 0.20} | s=0.10 |

Adds ~60 more inference tasks.

**Total Tier 2: ~585 tasks** (many of which are fast — freq methods take seconds, Gibbs at p=10 takes under a minute).

### 4.3 SLURM Strategy for the Expanded Grid

**Array 1: Frequentist + Gibbs at p=10** (fast batch)
- Batch all freq methods (3) + Gibbs for p=10 configs into single tasks.
- 5 (γ values) × 1 task each = 5 tasks, 1 CPU, 4 GB, 30 min.

**Array 2: Gibbs at p=50 and p=100** (medium)
- One task per (config, seed).
- p=50: ~25 tasks × 30 min each. p=100: ~25 tasks × 3 hours each.
- 1 CPU, 8 GB, 4 hours.

**Array 3: ADVI (MF + LR)** (medium)
- One task per (config, seed, method).
- ~150 tasks, 1 CPU, 8 GB, 2 hours.

**Array 4: NUTS** (heavy)
- One task per (config, seed).
- p=10: fast (< 5 min). p=50: 80–120 min. p=100: likely timeout.
- ~75 tasks, 1 CPU, 16 GB, 4 hours for p≤50; 6 hours for p=100.

**Array 5: Frequentist at p=50 and p=100**
- Batch all 3 freq methods × all seeds per config.
- ~25 tasks, 1 CPU, 4 GB, 30 min.

### 4.4 Reusing Tier 1 Results

Tier 1 already has results for p=50, γ ∈ {0.67, 0.20}, ER, s=0.10, seeds 0–4 for NUTS, ADVI-MF, and all freq methods (20 seeds for freq). The inference script's `--skip-existing` flag should detect these and not re-run them. Only new runs needed:
- Gibbs for the Tier 1 configs (new method)
- ADVI-LR for the Tier 1 configs (new method)
- All methods for the remaining (p, γ) combinations
- (If doing s=0.05: all methods for s=0.05 configs)

---

## 5. Progress Report (April 15)

### 5.1 Content (6 pages)

**Pages 1–2: Introduction + Model + Methods** — research question, graphical horseshoe model, the 7 methods (note Gibbs as "in progress"), evaluation metrics, implementation notes. Briefly note that sparsity s=0.10 is the default throughout, but s=0.05 runs may be included if time permits.

**Pages 3–4: Preliminary Results** — Tier 1 tables and figures from WORK2 §6–7. Two figures:
- Figure 1: Heatmap comparison (6 panels)
- Figure 2: Shrinkage profile comparison (NUTS vs ADVI-MF κ̂)

Five findings condensed into ~4 paragraphs. Highlight the bimodality surprise.

**Page 5: Remaining Work**

| Step | Deadline | Owner |
|------|----------|-------|
| Implement Li et al. Gibbs sampler + validate | April 20 | Nick |
| Expand to all p ∈ {10, 50, 100} × 5 γ values | April 22 | Arturo, Nick |
| Low-rank ADVI + PSIS diagnostics | April 24 | Federico |
| CRSP real data acquisition + preprocessing | April 24 | Federico |
| Sensitivity: block-diagonal + sparsity levels (including s=0.05 if time allows) | April 25 | Arturo |
| Real data inference + portfolio backtest | April 28 | All |
| Final report draft | May 1 | All |
| Figures, editing, presentation | May 3 | All |

**Page 5 (cont.): Plot Description** — Stein's loss vs. γ plot with 7 lines (one per method), shaded ±1 std bands. Explain what the reader learns: the critical γ at which Bayesian methods dominate, and whether Gibbs tracks NUTS or diverges. Consider adding a panel or annotation to illustrate the difference between s=0.10 and s=0.05 if the latter is run.

**Page 6: References.**

### 5.2 Figures to Generate Now

Use `scripts/generate_figures.py` on Tier 1 data. Config 45 (p=50, γ=0.20), seed 0 for the heatmap. Both configs for the shrinkage profile. If s=0.05 runs are available, generate one additional summary plot to highlight sparsity effects.

---

## 6. CRSP Real Data Analysis

### 6.1 Data Acquisition

Pull from WRDS (CRSP daily stock file). Select p=50 most liquid large-cap stocks. T=250 (1 year, γ=0.20) and optionally T=75 (γ=0.67). Demean and standardize.

### 6.2 Methods to Run

All 7 methods. On real data, the Gibbs sampler is particularly interesting because it may be the only exact MCMC option if NUTS is too slow for the real dataset's dimensionality.

### 6.3 Evaluation Without Ground Truth

- Out-of-sample log-likelihood (train/test split)
- Cross-method Frobenius distances
- GICS sector overlay on estimated graph
- GMV portfolio out-of-sample volatility
- PSIS-k̂ for ADVI methods

### 6.4 Real Data Figures

- Figure 7: Network graph from NUTS (or Gibbs if NUTS is slow), colored by GICS sector
- Figure 8: OOS portfolio volatility bar chart across methods

---

## 7. Final Report Structure (12 pages)

### Section 1: Introduction (1 page)
Research question. Preview the surprise finding. Preview the Gibbs vs. NUTS comparison.

### Section 2: Model (1.5 pages)
Graphical horseshoe. Non-centered parameterization. Shrinkage coefficient κ and its bimodal property.

### Section 3: Methods (2.5 pages)
- NUTS: configuration, convergence diagnostics.
- Gibbs: column-wise decomposition, data-augmentation trick, truncated normal rejection. Contrast with NUTS (model-specific vs. generic; component-wise vs. joint proposals).
- ADVI: mean-field + low-rank. The stability stack. PSIS diagnostics.
- Frequentist benchmarks.
- Evaluation metrics: Stein's loss, sparsity recovery, coverage, bimodality, PSIS-k̂.

### Section 4: Experiments — Synthetic (3 pages)
- **Table 1**: Full results across all (p, γ) for the 7 methods.
- **Figure 1**: Heatmap comparison.
- **Figure 2**: Shrinkage profiles (NUTS vs Gibbs vs ADVI-MF).
- **Figure 3**: Stein's loss vs. γ curves with all 7 methods.
- **Figure 4**: F1 vs. γ curves.
- **Figure 5**: Runtime vs. p (bar chart showing Gibbs fills the gap between fast ADVI and slow NUTS).
- Discussion: bimodality surprise, calibration gap, Gibbs vs NUTS quality, PSIS values. If s=0.05 results are available, add separate comparison or subpanel to show sparsity impact.

### Section 5: Experiments — Real Data (1.5 pages)
- Graph visualization, portfolio metrics, OOS log-likelihood.

### Section 6: Discussion (1.5 pages)
- Practitioner decision tree: use NUTS if p < 50 and time allows; Gibbs if p ≤ 100 and you can derive conditionals; ADVI if p > 100 or time is limited; glasso if no Bayesian infrastructure.
- The bimodality finding: ADVI preserves between-entry shrinkage structure but not within-entry posterior shape. What does this mean for the horseshoe's practical robustness?
- PSIS-k̂ as a standalone VI diagnostic.
- If s=0.05 results are included, discuss implications of the ultra-sparse scenario: does the horseshoe's theoretical advantage over glasso materialize empirically at this sparsity?

### Section 7: Conclusion (0.5 pages)
Summary, limitations, future work (normalizing flows, time-varying models, multivariate t-likelihood).

---

## 8. Key Figures (Priority-Ordered)

| Priority | Figure | Section | What it shows |
|----------|--------|---------|--------------|
| **Must** | Stein's loss vs. γ (7 methods, 3 panels for p=10/50/100) | §4 | Central result |
| **Must** | Shrinkage profiles (NUTS vs Gibbs vs ADVI-MF) | §4 | Bimodality hypothesis |
| **Must** | Heatmap comparison (7 panels now) | §4 | Visual intuition |
| **Must** | Cross-method metrics table | §4 | Quantitative summary |
| **Must** | Runtime vs. p (showing Gibbs fills the gap) | §4 | Practical guidance |
| **Must** | Real data graph visualization | §5 | Application |
| **Should** | F1 vs. γ curves | §4 | Sparsity recovery |
| **Should** | Coverage comparison (NUTS vs Gibbs vs ADVI-MF vs ADVI-LR) | §4 | Calibration |
| **Should** | PSIS-k̂ by method and γ | §4 | VI diagnostic |
| **Nice** | Sparsity sensitivity (s ∈ {0.05, 0.10, 0.30}) | §4 | Robustness; pay special attention to s = 0.05 if run — this is where the horseshoe should have the largest relative advantage over glasso |
| **Nice** | ER vs block-diagonal | §4 | Structure sensitivity |

---

## 9. Implementation Tasks

### 9.1 New Code

| File | Purpose | Est. Lines | Priority |
|------|---------|-----------|----------|
| `src/inference/gibbs_runner.py` | Li et al. Gibbs sampler | 250–350 | **P0** (critical path) |
| `src/evaluation/psis.py` | PSIS-k̂ computation | 60–100 | P1 |
| `scripts/generate_figures.py` | All publication figures | 500–700 | P1 |
| `scripts/aggregate_results.py` | Summary tables | 200–300 | P1 |
| `src/data/wrds_pull.py` | CRSP acquisition | 80–120 | P2 |
| `src/data/preprocess.py` | Demean, standardize, windows | 100–150 | P2 |
| `src/portfolio/gmv.py` | Weights, variance, backtest | 150–200 | P2 |
| `src/evaluation/oos_metrics.py` | OOS log-likelihood | 80–100 | P2 |
| `paper/final_report.tex` | LaTeX report | 1200–1500 | P1 |
| `paper/progress_report.tex` | LaTeX progress report | 400–600 | **P0** (due Apr 15) |
| `paper/presentation.tex` | Beamer slides | 300–400 | P3 |

### 9.2 Existing Code to Extend

| File | Change |
|------|--------|
| `src/inference/run_single.py` | Add `gibbs` dispatch, `advi_lr` dispatch, `--skip-existing` flag |
| `src/inference/advi_runner.py` | Add `low_rank` guide type |
| `src/evaluation/evaluate_single.py` | Add PSIS computation for ADVI methods |
| `src/evaluation/metrics.py` | Add `oos_log_likelihood`, `portfolio_metrics` |
| `scripts/generate_task_manifests.py` | Add `gibbs` and `advi_lr` to method lists, `--tier 2` support |

### 9.3 Tests to Add

| Test | What it verifies |
|------|-----------------|
| `test_gibbs_smoke_p5` | Gibbs sampler converges on p=5, T=500 toy problem |
| `test_gibbs_pd_maintained` | Ω stays PD throughout the chain |
| `test_gibbs_matches_nuts_p5` | Gibbs posterior mean ≈ NUTS posterior mean on toy problem (within sampling noise) |
| `test_gibbs_shrinkage_computed` | κ samples are correctly derived from λ and τ samples |
| `test_advi_lr_smoke_p5` | Low-rank ADVI converges |
| `test_psis_known_good` | PSIS-k̂ < 0.5 when q ≈ p |
| `test_psis_known_bad` | PSIS-k̂ > 0.7 when q is mismatched |
| `test_sparsity_effect_s005` | (If s=0.05 is run) The horseshoe prior should outperform glasso on very sparse data |

---

## 10. Timeline

### Week 1: April 12–18

| Day | Task | Owner |
|-----|------|-------|
| Apr 12–13 | Aggregate Tier 1, generate Figs 1–2, draft progress report | All |
| Apr 14 | Finalize and proofread progress report | All |
| **Apr 15** | **Submit progress report** | All |
| Apr 15–17 | **Implement Gibbs sampler** (`gibbs_runner.py`) | Nick |
| Apr 16–17 | Validate Gibbs on p=5 toy (must match NUTS posterior) | Nick |
| Apr 17 | Add `advi_lr` dispatch + PSIS computation | Federico |
| Apr 18 | Submit Tier 2 freq + Gibbs p=10 arrays | Arturo |

### Week 2: April 19–25

| Day | Task | Owner |
|-----|------|-------|
| Apr 19 | Submit Tier 2 NUTS + Gibbs p=50 arrays | Arturo |
| Apr 19–20 | Submit Tier 2 ADVI (MF + LR) arrays | Federico |
| Apr 20 | Submit Tier 2 Gibbs + ADVI + freq p=100 arrays | Arturo |
| Apr 20 | Submit Tier 2 NUTS p=100 (expect timeouts) | Arturo |
| Apr 21 | CRSP data acquisition + preprocessing | Federico |
| Apr 22 | Audit Tier 2, resubmit failures | All |
| Apr 23 | Run all 7 methods on CRSP data | All |
| Apr 24 | Sensitivity: block-diagonal + sparsity levels (run s=0.05 if possible) | Arturo |
| Apr 25 | Aggregate all results, build summary tables + loss-vs-γ data; include sparsity sensitivity plots if ready | Arturo |

### Week 3: April 26 – May 5

| Day | Task | Owner |
|-----|------|-------|
| Apr 26 | Generate all publication figures | Arturo |
| Apr 27 | Portfolio analysis (synthetic + real) | Federico |
| Apr 28 | Draft §1–3 (intro, model, methods) | Nick |
| Apr 29 | Draft §4 (synthetic experiments) | Arturo |
| Apr 30 | Draft §5–6 (real data, discussion) | Federico |
| May 1 | Integrate, edit, check page count | All |
| May 2 | Internal review, revise | All |
| May 3 | Build presentation slides (12 slides) | All |
| May 4 | Final proofread, figure polish, reference check | All |
| **May 5** | **Submit final report** | All |

---

## 11. Risk Registry

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Gibbs sampler truncated-normal rejection rate high at p=100 | Medium | Use Schur complement PD check (O(p²) not O(p³)). If still bad, try ESS-based rejection (elliptical slice sampling for the truncated normal). |
| Gibbs sampler ESS very low (slow mixing) | Medium | This is a finding. Report it. Increase thinning. The comparison with NUTS ESS/gradient is still informative. |
| NUTS times out at p=100 | Expected | Gibbs fills the gap. Report NUTS timeout as a finding. |
| PSIS requires evaluating log q(θ) which is tricky for AutoLowRankMVN | Low | ArviZ may handle this automatically. If not, use a simpler estimate: compute log-weights only for AutoNormal (MF) where log q is trivial. |
| CRSP data acquisition fails | Medium | Fall back to yfinance or Kenneth French 49 industry portfolios. |
| Report exceeds 12 pages | Medium | ICML two-column saves ~30%. Move Gibbs algorithm details and NUTS convergence tables to appendix. |
| Cluster congestion | Medium | Submit all heavy jobs by Apr 20. Keep 1-week buffer before writing. |
| **s=0.05 sparsity compute fits** | Low/Medium | Prioritize one s=0.05 setting if short on compute. If unable, explain in report. Partial results still valuable. |

---

## 12. What NOT to Do

- **Do not implement the Gibbs sampler in JAX.** Use plain NumPy. The Gibbs sampler is inherently sequential (each column update depends on the previous one), so JAX's compilation/vectorization provides little benefit. NumPy is simpler to debug and the O(p³) linear algebra is already fast in LAPACK.
- **Do not run the Gibbs sampler with 4 chains.** One long chain is the correct approach for a single-site Gibbs sampler. Use Geweke + ESS for convergence assessment instead of R̂.
- **Do not skip the Gibbs validation on p=5.** If the Gibbs posterior mean doesn't match NUTS's posterior mean on a trivial problem, there's a bug in the conditionals.
- **Do not compute PSIS for NUTS or Gibbs.** PSIS is a VI diagnostic. MCMC methods have their own diagnostics (R̂, ESS, divergences, Geweke).
- **Do not over-invest in the portfolio backtest.** A single-date GMV comparison is sufficient for both synthetic and real data. The rolling backtest is a nice-to-have for the final report, not a must-have.
- **Do not pursue normalizing flows or SVGD.** Out of scope for 3 weeks.
- **Do not rerun Tier 1 results.** Use `--skip-existing`.
- **Do not attempt s=0.05 everywhere if under deadline.** It's valid to run s=0.05 only for one (p, γ) if compute/time limited, and note this in the report.

---

## 13. Deliverables Checklist

### Progress Report (April 15)
- [ ] 6-page PDF emailed to staff
- [ ] Figure 1: Heatmap comparison
- [ ] Figure 2: Shrinkage profiles
- [ ] Table 1: Cross-method metrics
- [ ] Remaining work plan with Gibbs + PSIS
- [ ] Plot description for loss-vs-γ
- [ ] (Optional, if done) Figure/table for s=0.05 sparsity

### Gibbs Sampler (April 20)
- [ ] `src/inference/gibbs_runner.py` — complete implementation
- [ ] Passes `test_gibbs_smoke_p5` and `test_gibbs_matches_nuts_p5`
- [ ] Integrated into `run_single.py` dispatcher
- [ ] Runs on Tier 1 configs (p=50, 2 γ values, 5 seeds)

### Tier 2 Results (April 25)
- [ ] All 7 methods × 15 (p, γ) pairs × 5 seeds
- [ ] PSIS-k̂ computed for all ADVI runs
- [ ] Sensitivity: s ∈ {0.05, 0.30} and block-diagonal (s=0.05 prioritized if time allows)
- [ ] Aggregated summary tables
- [ ] Loss-vs-γ curves for all 3 p values
- [ ] (If run) Additional plot showing s=0.05 effect

### Real Data (April 27)
- [ ] CRSP data acquired and preprocessed
- [ ] All 7 methods run on p=50, T=250
- [ ] Graph visualization
- [ ] OOS metrics

### Final Report (May 5)
- [ ] 12-page PDF, ICML style
- [ ] All must-have figures
- [ ] Division of labor stated
- [ ] References complete

### Presentation (early May)
- [ ] 12 slides, 10–15 min
- [ ] Backup slides for Q&A