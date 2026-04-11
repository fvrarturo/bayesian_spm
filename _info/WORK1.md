# Agent Prompt: Synthetic Data Generation Pipeline for Sparse Bayesian Precision Matrix Estimation

## Context

You are building the **synthetic data generation module** for a research project comparing MCMC (NUTS) vs. Variational Inference (ADVI) for sparse Bayesian precision matrix estimation using the graphical horseshoe prior. This is for MIT course 6.7830 (Bayesian Modeling and Inference, Spring 2026).

The project compares how well different inference methods recover a **known** sparse precision matrix Ω₀ from data Y sampled from N(0, Ω₀⁻¹). The synthetic data generation is foundational: every downstream comparison (loss metrics, sparsity recovery, shrinkage profiles, portfolio construction) depends on having well-constructed ground-truth precision matrices and corresponding data.

This code will run on the **MIT Engaging cluster** (SLURM scheduler, `mit_normal` partition: up to 16 CPUs/job, 128 GB RAM, 12-hour wall time). Data generation is embarrassingly parallel — each (config, seed) pair is independent — so we will use **SLURM job arrays**.

---

## 1. What You Are Building

A pipeline that, for every combination of experimental parameters, does two things:

1. **Generates a sparse, symmetric, positive-definite precision matrix Ω₀** with a known sparsity pattern (the "ground truth").
2. **Samples T i.i.d. observations** Y = [y₁, …, y_T]ᵀ from N(0, Ω₀⁻¹), producing a (T × p) data matrix.

Each (Ω₀, Y) pair, along with metadata, is saved to disk in a structured directory so that inference scripts can later load them by config ID.

---

## 2. The Experimental Grid

### 2.1 Dimensions Being Varied

| Factor | Levels | Rationale |
|--------|--------|-----------|
| **p** (dimension) | 10, 50, 100 | p=10 is for debugging/smoke tests. p=50 is the main experimental setting. p=100 tests scalability. |
| **γ = p/T** (concentration ratio) | 0.90, 0.67, 0.42, 0.20, 0.10 | Controls estimation difficulty. γ→1 means the sample covariance is nearly singular and regularization is essential. γ=0.10 is the "easy" regime where even naive methods work. |
| **Graph structure** | Erdős–Rényi, Block-diagonal | ER is unstructured random sparsity. Block-diagonal mimics sector structure in financial data (stocks within the same industry are conditionally dependent, but independent across sectors). |
| **s** (sparsity) | 0.05, 0.10, 0.30 | s = fraction of possible edges present. s=0.05 is very sparse (horseshoe should dominate). s=0.10 is moderate (our main case). s=0.30 is "mild" — still sparse but denser, testing where the horseshoe advantage narrows. |
| **Seed** | 0, 1, 2, …, 19 | 20 independent replications per configuration. Provides error bars and statistical reliability for all downstream metrics. |

### 2.2 Derived Sample Sizes

T is derived from p and γ as T = ⌈p / γ⌉:

| p \ γ | 0.90 | 0.67 | 0.42 | 0.20 | 0.10 |
|-------|------|------|------|------|------|
| **10** | 12 | 15 | 24 | 50 | 100 |
| **50** | 56 | 75 | 120 | 250 | 500 |
| **100** | 112 | 150 | 239 | 500 | 1000 |

### 2.3 Configurations to Skip

For **s = 0.30** (the denser setting), skip **γ = 0.90**. Rationale: with 30% of edges present, the true Ω₀ has many nonzero entries and the diagonal loading needed to ensure positive definiteness can be very large, distorting the matrix. Combined with γ = 0.90 (barely more observations than parameters), the estimation problem is pathologically hard and uninformative — no method will do well, so the comparison adds nothing. This removes 2 × 3 = 6 configs (2 graphs × 3 p-values).

### 2.4 Full Configuration Count

**Without the skip rule:**
- 3 (p) × 5 (γ) × 2 (graph) × 3 (s) = 90 configs

**After removing s=0.30, γ=0.90:**
- 90 − 6 = **84 unique configurations**

**With 20 seeds each:**
- 84 × 20 = **1,680 total simulations**

This is the upper bound. All 1,680 (Ω₀, Y, metadata) triples should be generated and stored. We can always use a subset for inference (which is the expensive part), but we never want to have to re-generate data.

---

## 3. How to Generate Each Component

### 3.1 Generating the Precision Matrix Ω₀

The precision matrix must satisfy three properties:
1. **Symmetric**: Ω₀ = Ω₀ᵀ
2. **Positive definite**: all eigenvalues > 0 (strictly: we enforce λ_min ≥ 0.1)
3. **Sparse**: only a fraction s of off-diagonal entries are nonzero, with a known edge set E₀

#### Erdős–Rényi Graph

1. Create a `networkx.erdos_renyi_graph(p, s, seed=seed)`.
2. For each edge (i, j) in the graph, draw a sign uniformly from {−1, +1} and a magnitude from Uniform(0.3, 0.8). Set ω_{ij} = ω_{ji} = sign × magnitude.
3. Set all diagonal entries to 1 initially.
4. Compute the minimum eigenvalue λ_min of the resulting matrix. If λ_min < 0.1, add (0.1 − λ_min) × I_p to the diagonal. This guarantees PD with a minimum eigenvalue of at least 0.1.

**Why [0.3, 0.8] for signal strength?** Values below 0.2 are effectively indistinguishable from zero for any method at realistic sample sizes, making edge recovery trivially hard for everyone. Values above 1.0 risk producing ill-conditioned matrices after diagonal loading. The range [0.3, 0.8] ensures edges are detectable signals.

#### Block-Diagonal Graph

1. Partition the p nodes into n_blocks groups of roughly equal size. Convention:
   - p=10: n_blocks = 2 (blocks of 5)
   - p=50: n_blocks = 5 (blocks of 10)
   - p=100: n_blocks = 5 (blocks of 20)
2. Within each block, add edges independently with probability `intra_sparsity = s` (i.e., the sparsity parameter s now controls within-block density).
3. **No edges between blocks.** This is the defining feature — it models sector structure.
4. Assign edge weights and ensure PD exactly as in the ER case.

**Important:** The overall sparsity for block-diagonal will be *lower* than s because edges can only exist within blocks, not between them. For p=50 with 5 blocks of 10, the number of within-block pairs is 5 × C(10,2) = 225 out of C(50,2) = 1225 total pairs. So global sparsity ≈ s × 225/1225 ≈ 0.184 × s. At s=0.10, global sparsity is only about 1.8%. This is fine — it means the block-diagonal structure is genuinely sparser, which is realistic for sector-structured data.

**Note on the last block**: when p is not divisible by n_blocks, the last block absorbs the remainder. For p=50, n_blocks=5, all blocks are exactly size 10. For p=100, n_blocks=5, all blocks are exactly size 20. For p=10, n_blocks=2, both blocks are size 5.

### 3.2 Sampling Data from Ω₀

Given a valid Ω₀:
1. Compute Σ₀ = Ω₀⁻¹ (use `numpy.linalg.inv` — the matrix is moderate-sized and well-conditioned by construction).
2. Draw Y ∈ ℝ^{T×p} where each row y_k ~ N(0, Σ₀) independently.
3. Use `numpy.random.default_rng(seed).multivariate_normal(zeros(p), Sigma_0, size=T)`.

**Critical**: the same seed that determines the graph structure should NOT be reused for sampling. Use a deterministic but distinct seed derivation, e.g., `graph_seed = seed`, `data_seed = seed + 10000`. This ensures that for a given config, changing the random seed changes both the graph and the data, but the two are generated from independent RNG streams.

### 3.3 Precomputed Metadata to Store

For each (Ω₀, Y) pair, also compute and store:

| Quantity | Why |
|----------|-----|
| `Sigma_0 = Ω₀⁻¹` | Needed for GMV oracle portfolio weights and for computing out-of-sample portfolio variance |
| `edge_set` | The set of (i, j) pairs with i < j where ω_{ij} ≠ 0. This is the ground truth for sparsity recovery (TPR, FPR, MCC, F1). |
| `eigenvalues` | Sorted eigenvalues λ₁ ≥ … ≥ λ_p of Ω₀. Used for eigenvalue spectrum comparison plots. |
| `condition_number` | κ(Ω₀) = λ₁/λ_p. Flags ill-conditioned ground truths. |
| `n_edges` | Number of true edges. Quick sanity check. |
| `gmv_weights_oracle` | w* = Ω₀ 1 / (1ᵀ Ω₀ 1). The "best possible" GMV portfolio for downstream comparison. |
| `oracle_portfolio_variance` | σ² = 1 / (1ᵀ Ω₀ 1). Lower bound on portfolio variance. |

---

## 4. Directory Structure and File Naming

### 4.1 Directory Layout

```
data/
  synthetic/
    configs/
      config_manifest.json          # Master list of all configurations
    erdos_renyi/
      p010/
        gamma090/
          s005/
            seed_00/
              omega_true.npy        # (p, p) float64
              sigma_true.npy        # (p, p) float64
              Y.npy                 # (T, p) float64
              metadata.json         # all scalar metadata + edge_set as list
            seed_01/
              ...
            ...
            seed_19/
          s010/
            ...
          s030/     # NOTE: absent for gamma090 (skipped config)
            ...
        gamma067/
          ...
        gamma042/
          ...
        gamma020/
          ...
        gamma010/
          ...
      p050/
        ...
      p100/
        ...
    block_diagonal/
      p010/
        ...
      p050/
        ...
      p100/
        ...
```

### 4.2 File Naming Conventions

- Gamma values encoded without dots: `gamma090`, `gamma067`, `gamma042`, `gamma020`, `gamma010`
- Sparsity encoded: `s005`, `s010`, `s030`
- Dimension encoded with zero-padding: `p010`, `p050`, `p100`
- Seeds zero-padded to 2 digits: `seed_00` through `seed_19`

### 4.3 The Config Manifest

Generate a single JSON file `config_manifest.json` that lists every valid configuration with a unique integer `config_id` (0 through 83). Each entry contains:

```json
{
  "config_id": 0,
  "p": 50,
  "gamma": 0.42,
  "T": 120,
  "graph": "erdos_renyi",
  "sparsity": 0.10,
  "n_blocks": null,
  "signal_range": [0.3, 0.8],
  "n_seeds": 20,
  "skip": false,
  "dir_path": "erdos_renyi/p050/gamma042/s010"
}
```

This manifest is the single source of truth. All downstream scripts (inference, evaluation) should load it to discover what data exists.

### 4.4 The metadata.json per seed

```json
{
  "config_id": 7,
  "p": 50,
  "T": 120,
  "gamma": 0.42,
  "graph": "erdos_renyi",
  "sparsity": 0.10,
  "seed": 3,
  "graph_seed": 3,
  "data_seed": 10003,
  "n_edges": 127,
  "n_possible_edges": 1225,
  "realized_sparsity": 0.1037,
  "condition_number": 14.32,
  "min_eigenvalue": 0.1,
  "max_eigenvalue": 1.432,
  "trace_omega": 62.71,
  "oracle_portfolio_variance": 0.00217,
  "edge_set": [[0, 1], [0, 7], [1, 3], ...],
  "diagonal_shift_applied": 0.73,
  "timestamp": "2026-04-11T14:30:00"
}
```

---

## 5. Validation and Sanity Checks

Every generated (Ω₀, Y) pair must pass the following checks before being saved. If any check fails, log the failure and skip that seed (do not save corrupted data).

### 5.1 Checks on Ω₀

1. **Symmetry**: `numpy.allclose(Omega, Omega.T, atol=1e-12)` → must be True.
2. **Positive definiteness**: all eigenvalues ≥ 0.1 (our construction guarantees this, but verify).
3. **Correct sparsity pattern**: the set of nonzero off-diagonal entries matches the edge set from the graph generator. No off-diagonal entry should be nonzero unless it corresponds to a graph edge. All graph edges should produce nonzero entries (i.e., none of the random weights are accidentally zero).
4. **Reasonable condition number**: κ(Ω₀) < 1000. If exceeded, log a warning (don't skip — high condition numbers are informative, but they should be flagged).
5. **Diagonal loading was not extreme**: if the diagonal shift exceeds 2.0, log a warning. Large shifts mean the random graph + weights almost produced a non-PD matrix, and the resulting Ω₀ may have artificially inflated diagonal entries.

### 5.2 Checks on Y

1. **Shape**: Y.shape == (T, p).
2. **Finite**: no NaN or Inf values.
3. **Approximate zero mean**: `numpy.abs(Y.mean(axis=0)).max()` < 3/√T (3-sigma tolerance for sampling noise around zero).
4. **Sample covariance rank**: `numpy.linalg.matrix_rank(numpy.cov(Y, rowvar=False))` should equal min(T−1, p). This confirms the data has the expected rank structure.

### 5.3 Checks on Σ₀ = Ω₀⁻¹

1. **Reconstruction**: `numpy.allclose(Omega @ Sigma, numpy.eye(p), atol=1e-8)` → must be True.
2. **PD**: all eigenvalues of Σ₀ are positive (follows from Ω₀ being PD, but verify numerically).

---

## 6. Implementation Architecture

### 6.1 Core Functions to Implement

```
src/utils/matrix_utils.py:
  - sparse_omega_erdos_renyi(p, sparsity, signal_range, seed) → (Omega, edge_set)
  - sparse_omega_block_diagonal(p, n_blocks, intra_sparsity, signal_range, seed) → (Omega, edge_set)
  - _graph_to_omega(G, p, signal_range, rng) → (Omega, edge_set)   [private helper]
  - assemble_precision_matrix(omega_offdiag, omega_diag, p) → Omega  [used by inference code too]
  - sample_data_from_omega(Omega, T, seed) → Y

src/utils/validation.py:
  - validate_omega(Omega, edge_set, p) → (is_valid, diagnostics_dict)
  - validate_data(Y, T, p) → (is_valid, diagnostics_dict)

scripts/generate_synthetic_data.py:
  - generate_single_config(config_dict, seed, output_dir) → success/failure
  - generate_all_configs(manifest_path, output_base_dir) → summary

scripts/generate_config_manifest.py:
  - build_manifest() → writes config_manifest.json
```

### 6.2 Single-Config Generation Function (Pseudocode)

```
def generate_single_config(config, seed, output_dir):
    1. Determine graph_seed = seed, data_seed = seed + 10000
    2. If config["graph"] == "erdos_renyi":
         Omega, edge_set = sparse_omega_erdos_renyi(p, s, signal_range, graph_seed)
       Elif config["graph"] == "block_diagonal":
         Omega, edge_set = sparse_omega_block_diagonal(p, n_blocks, s, signal_range, graph_seed)
    3. Validate Omega. If invalid, log and return failure.
    4. Compute Sigma = inv(Omega)
    5. Validate Sigma. If invalid, log and return failure.
    6. Y = sample_data_from_omega(Omega, T, data_seed)
    7. Validate Y. If invalid, log and return failure.
    8. Compute metadata: eigenvalues, condition_number, n_edges, gmv_weights, etc.
    9. Save Omega, Sigma, Y as .npy; metadata as .json
    10. Return success
```

### 6.3 Parallelization Strategy for the Cluster

Each of the 1,680 simulations is independent and takes seconds to run (even p=100, T=1000 is trivial — it's a matrix inversion + multivariate normal sample). The bottleneck is not compute but I/O and job scheduling overhead.

**Recommended approach: SLURM job array with batching.**

Rather than submitting 1,680 individual jobs (too much scheduler overhead), batch multiple configs per job:

- **Each SLURM array task processes all 20 seeds for one configuration** (i.e., one (p, γ, graph, s) combo).
- Array size: 84 tasks (one per config).
- Each task is single-core, takes < 2 minutes, needs < 4 GB RAM.
- Request 1 CPU, 4 GB RAM, 15 minutes wall time per task.
- Partition: `mit_normal`

```bash
#!/bin/bash
#SBATCH --job-name=gen_synth
#SBATCH --array=0-83
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err

module load miniforge
conda activate ggm_horseshoe

python scripts/generate_synthetic_data.py \
    --config-id $SLURM_ARRAY_TASK_ID \
    --manifest data/synthetic/configs/config_manifest.json \
    --output-dir data/synthetic/ \
    --n-seeds 20
```

This submits all 84 configs in one `sbatch` command. SLURM handles scheduling. The whole generation finishes in minutes once jobs start.

---

## 7. Storage Budget

| p | T (max) | Size of Y (max) | Size of Ω₀ | Size of Σ₀ | Total per seed |
|---|---------|-----------------|------------|------------|---------------|
| 10 | 100 | 8 KB | 0.8 KB | 0.8 KB | ~10 KB |
| 50 | 500 | 200 KB | 20 KB | 20 KB | ~240 KB |
| 100 | 1000 | 800 KB | 80 KB | 80 KB | ~960 KB |

**Worst case**: all 1,680 simulations at max sizes ≈ **1.6 GB total**. This is negligible on the cluster.

---

## 8. Testing Strategy

Before running on the cluster, validate locally:

1. **Unit tests** (`tests/test_synthetic.py`):
   - `test_erdos_renyi_is_pd`: Ω₀ is positive definite for p ∈ {10, 20, 50}.
   - `test_erdos_renyi_is_symmetric`: Ω₀ = Ω₀ᵀ.
   - `test_block_diagonal_is_pd`: same for block structure.
   - `test_block_diagonal_no_inter_block_edges`: verify no edges cross block boundaries.
   - `test_erdos_renyi_sparsity_in_range`: realized edge fraction is within [0.5s, 2s] (stochastic, so use tolerance).
   - `test_data_shape`: Y has shape (T, p).
   - `test_data_finite`: no NaN/Inf.
   - `test_sample_cov_rank`: sample covariance has rank min(T−1, p).
   - `test_reconstruction`: Ω₀ × Σ₀ ≈ I.
   - `test_gmv_weights_sum_to_one`: oracle portfolio weights sum to 1.
   - `test_config_manifest_count`: manifest has exactly 84 entries.
   - `test_no_skipped_configs_generated`: configs with s=0.30, γ=0.90 are absent.

2. **Smoke test** (local, before cluster):
   - Generate all 84 configs with n_seeds=1 (just seed 0).
   - Verify all 84 directories exist with the right files.
   - Check that the metadata.json files are parseable and consistent with the manifest.

3. **Post-generation audit** (after cluster run):
   - Script that walks the output directory, counts files, checks that every expected (config, seed) pair exists, and flags any missing or corrupted files.

---

## 9. Important Design Decisions and Their Rationale

### 9.1 Why 20 seeds?

With 20 replications, the standard error of the mean of any metric is σ/√20 ≈ 0.22σ. This gives tight enough error bars for publication-quality plots. 5 seeds (as in the original action plan) gives σ/√5 ≈ 0.45σ — error bars twice as wide, which can obscure real differences between methods. Since generation is free (seconds per sim), 20 is the right choice. We can always choose to run inference on fewer seeds if compute is the bottleneck.

### 9.2 Why store Σ₀ separately?

Computing Σ₀ = Ω₀⁻¹ is cheap, but it's needed frequently (portfolio variance, data generation verification) and we don't want to recompute it in every downstream script. At 80 KB for p=100, the storage cost is negligible.

### 9.3 Why the signal range [0.3, 0.8]?

This is the regime where the comparison is most informative. Below 0.3, edges are too weak to detect with any method at realistic T — the comparison degenerates to "everyone fails equally." Above 0.8, edges are so strong that even crude methods detect them, and the horseshoe's advantage disappears. The range [0.3, 0.8] sits in the sweet spot where method choice matters.

### 9.4 Why only two graph structures?

Erdős–Rényi is the standard unstructured benchmark used in the graphical horseshoe paper (Li et al. 2019) and most GGM literature. Block-diagonal is the most natural structured alternative for our financial application (it directly mimics sector/industry groupings). Other structures (star, scale-free, small-world, band) are interesting but add complexity without changing the core narrative. If time permits, a band structure can be added later — the pipeline is designed to be extensible.

### 9.5 Why enforce minimum eigenvalue 0.1 specifically?

The diagonal loading shifts all eigenvalues up by the same constant. Setting the floor at 0.1 ensures Ω₀ is numerically well-conditioned enough for:
- Stable inversion (Σ₀ = Ω₀⁻¹ is accurate)
- Stable Cholesky decomposition (needed inside NumPyro's MultivariateNormal)
- Reasonable condition numbers (κ < ~100 for most configs)

A floor of 0.01 would be too aggressive (condition numbers in the thousands), and 1.0 would be too conservative (would over-inflate the diagonal, weakening off-diagonal signals).

---

## 10. What NOT to Do

- **Do not demean or standardize Y.** The data is generated with true zero mean. Demeaning introduces unnecessary noise. Standardization would destroy the scale information in Ω₀. Preprocessing is a separate step that happens before inference, not during data generation.
- **Do not hardcode paths.** All paths should derive from the manifest + a base output directory.
- **Do not use a different RNG for each function call** without explicit seeding. Every random operation must be reproducible from the (config_id, seed) pair alone.
- **Do not store the data in any format other than .npy** for arrays. It's fast, compact, and universally readable in Python. JSON for metadata only.
- **Do not skip the validation checks.** A corrupted Ω₀ (e.g., not actually PD due to a floating-point edge case) will cause NUTS to crash with cryptic errors days later. Catch it now.
- **Do not generate data inside the inference scripts.** Data generation and inference are strictly separated. The inference script receives a path to a pre-generated (Ω₀, Y) directory and does nothing else.

---

## 11. Deliverables Checklist

When this module is complete, the following should exist and be verified:

- [ ] `scripts/generate_config_manifest.py` — produces `config_manifest.json` with 84 entries
- [ ] `src/utils/matrix_utils.py` — contains all generators + `assemble_precision_matrix`
- [ ] `src/utils/validation.py` — contains all validation functions
- [ ] `scripts/generate_synthetic_data.py` — CLI tool that generates data for one config_id (all seeds)
- [ ] `scripts/generate_synthetic_slurm.sh` — SLURM array job script
- [ ] `scripts/audit_synthetic_data.py` — post-generation audit script
- [ ] `tests/test_synthetic.py` — pytest suite (all tests pass)
- [ ] `data/synthetic/configs/config_manifest.json` — 84 configs
- [ ] `data/synthetic/` — 1,680 seed directories, each with `omega_true.npy`, `sigma_true.npy`, `Y.npy`, `metadata.json`
- [ ] All validations pass; audit script reports 0 missing or corrupted files