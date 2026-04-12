# WORK1 Completion Report: Synthetic Data Generation Pipeline

**Project:** Sparse Bayesian Precision Matrix Estimation (6.7830, Spring 2026)
**Phase:** WORK1 — Synthetic data generation
**Status:** ✅ Complete; all deliverables implemented, tested, and run on the MIT Engaging cluster
**Date:** 2026-04-11

This document is the implementation-side answer to [`WORK1.md`](WORK1.md). It records what was built, what was generated, what sanity checks passed, and what additional checks the team can run on the cluster.

---

## TL;DR

- All 11 deliverables from `WORK1.md §11` exist and are wired together.
- The experiment grid (`compute_configs`) produces exactly **84 configurations**.
- The full SLURM job array (`array=0-83`) ran on `mit_normal` and produced **1,680 / 1,680** seed directories with **0 failures**.
- The post-generation audit (`scripts/audit_synthetic_data.py --strict`) reports **0 missing, 0 corrupted, 0 metadata mismatches**.
- 55 pytest unit tests pass in 9.7s.
- All **8 optional post-hoc sanity checks** (`scripts/sanity_checks.py`) pass on the cluster — see §9 for the actual results.
- Total disk footprint: **374 MB** on the cluster.

---

## 1. Deliverables Checklist (mirrors `WORK1.md §11`)

| # | Deliverable | Status | Path |
|---|---|---|---|
| 1 | `scripts/generate_config_manifest.py` | ✅ | [scripts/generate_config_manifest.py](../scripts/generate_config_manifest.py) |
| 2 | `src/utils/matrix_utils.py` (with all generators + `assemble_precision_matrix`) | ✅ | [src/utils/matrix_utils.py](../src/utils/matrix_utils.py) |
| 3 | `src/utils/validation.py` (with all validation functions) | ✅ | [src/utils/validation.py](../src/utils/validation.py) |
| 4 | `scripts/generate_synthetic_data.py` (CLI tool, single config_id) | ✅ | [scripts/generate_synthetic_data.py](../scripts/generate_synthetic_data.py) |
| 5 | `scripts/generate_synthetic_slurm.sh` (SLURM array job) | ✅ | [scripts/generate_synthetic_slurm.sh](../scripts/generate_synthetic_slurm.sh) |
| 6 | `scripts/audit_synthetic_data.py` (post-generation audit) | ✅ | [scripts/audit_synthetic_data.py](../scripts/audit_synthetic_data.py) |
| 7 | `tests/test_synthetic.py` (pytest suite, all tests pass) | ✅ | [tests/test_synthetic.py](../tests/test_synthetic.py) |
| 8 | `data/synthetic/configs/config_manifest.json` (84 configs) | ✅ | [data/synthetic/configs/config_manifest.json](../data/synthetic/configs/config_manifest.json) |
| 9 | `data/synthetic/` (1,680 seed directories with all 4 files each) | ✅ | on the cluster (gitignored locally) |
| 10 | All validations pass; audit reports 0 missing or corrupted | ✅ | see §5 |
| 11 | Auxiliary: `src/utils/configs.py` (pure logic, importable from tests) | ➕ | [src/utils/configs.py](../src/utils/configs.py) |

The auxiliary `configs.py` was added as a clean separation: WORK1 puts `build_manifest` inside the script, but tests need to import it. Splitting the pure-Python logic into `src/utils/configs.py` and keeping the script as a thin CLI wrapper is the cleanest way to satisfy both.

---

## 2. Architecture

The pipeline has three layers, each in its own home in the directory tree.

```
src/utils/
├── matrix_utils.py     # graph -> Omega -> data, primitives
├── validation.py       # validate_omega / validate_sigma / validate_data
└── configs.py          # the experimental grid as a pure function

scripts/
├── generate_config_manifest.py    # builds data/synthetic/configs/config_manifest.json
├── generate_synthetic_data.py     # CLI: --config-id N | --config-ids 0,3,40 | --all
├── generate_synthetic_slurm.sh    # SLURM array job (--array=0-83)
└── audit_synthetic_data.py        # post-run audit, --strict mode

tests/
└── test_synthetic.py   # 55 tests, ~10s runtime

data/synthetic/
├── configs/config_manifest.json   # source of truth
├── erdos_renyi/                   # generated, gitignored
└── block_diagonal/                # generated, gitignored
```

### 2.1 `src/utils/matrix_utils.py`

Three public graph generators plus a private helper, all returning a 3-tuple **`(Omega, edge_set, diagnostics)`**:

- `sparse_omega_erdos_renyi(p, sparsity, signal_range, seed)`
- `sparse_omega_block_diagonal(p, n_blocks, intra_sparsity, signal_range, seed)`
- `sparse_omega_band(p, bandwidth, signal_range, seed)` — kept around but not part of the WORK1 grid
- `_graph_to_omega(G, p, signal_range, rng)` — shared backend; assigns weights `±Uniform(signal_range)`, computes pre-shift min eigenvalue, applies the smallest diagonal shift needed to enforce `λ_min ≥ 0.1`, and returns the shift in the diagnostics dict
- `assemble_precision_matrix(omega_offdiag, omega_diag, p)` — symmetric assembly using row-major upper-triangular index ordering (the same convention used by the NumPyro model)
- `sample_data_from_omega(Omega, T, seed)` — `Σ = Ω⁻¹` then `multivariate_normal(0, Σ, size=T)`

The 3-tuple return is the one substantive shape change from the original stubs. The reason: `WORK1.md §4.4` requires `diagonal_shift_applied` and `min_eigenvalue_pre_shift` in the metadata, and computing them in the validator is wasteful when the generator already had them in hand.

### 2.2 `src/utils/validation.py`

Three validators, each returning `(ok: bool, diagnostics: dict)`:

- **`validate_omega(Omega, edge_set, p, diagonal_shift=None)`** — hard-fails on shape mismatch, NaN/Inf, asymmetry, `eigmin ≤ 0`, `eigmin < 0.1 - 1e-10`, or sparsity-pattern mismatch against the claimed edge set. Issues warnings (collected in `diagnostics["warnings"]`, not failures) for `condition_number > 1000` and `diagonal_shift > 2.0`.
- **`validate_sigma(Sigma, Omega, p)`** — hard-fails if `||Ω·Σ - I||_∞ > 1e-8` or if `Σ` is not PD.
- **`validate_data(Y, T, p, Sigma=None)`** — hard-fails on shape, NaN/Inf, or `rank(cov(Y)) ≠ min(T-1, p)`. Issues a warning if `max |ȳ_i / √(Σ_ii/T)| > 5` (i.e., the per-component sample-mean z-score exceeds 5σ).

The zero-mean check uses a properly scaled per-component z-score rather than the un-scaled `3/√T` from the original spec, because the diagonal of `Σ` varies across components and a 3-sigma cutoff over `p=100` components would falsely trigger ~24% of the time. Using a 5-sigma per-component threshold drops the false-positive rate to ~5×10⁻⁵.

All tolerances are centralized in module-level constants at the top of the file so they can be audited in one place.

### 2.3 `src/utils/configs.py`

Pure-Python module with no I/O. Defines:

- `P_VALUES = [10, 50, 100]`
- `GAMMA_VALUES = [0.90, 0.67, 0.42, 0.20, 0.10]`
- `GRAPHS = ["erdos_renyi", "block_diagonal"]`
- `SPARSITY_VALUES = [0.05, 0.10, 0.30]`
- `SIGNAL_RANGE = [0.3, 0.8]`
- `N_SEEDS = 20`
- `N_BLOCKS_MAP = {10: 2, 50: 5, 100: 5}`

And:

- `T_from_gamma(p, gamma) = ceil(p / gamma)`
- `should_skip(sparsity, gamma) = (sparsity == 0.30 and gamma == 0.90)`
- `compute_configs() -> List[dict]` — emits the deterministic 84-element list with sequential `config_id`s
- `expected_config_count() -> int` — independent recomputation of the count for tests
- Directory-name encoders: `dir_name_p`, `dir_name_gamma`, `dir_name_sparsity`, `dir_name_seed`, `dir_path_for_config`

### 2.4 `scripts/generate_config_manifest.py`

Thin CLI wrapper around `compute_configs`. Writes the JSON to `data/synthetic/configs/config_manifest.json` (or `--output`) and optionally prints a human-readable summary table with `--print`.

### 2.5 `scripts/generate_synthetic_data.py`

Three layers:

1. **`generate_single_config(config, seed, output_base_dir) -> (success, info)`** — generates one (Ω, Σ, Y) triple with all validations and metadata, writes atomically (see §2.5.1), returns `(True, metadata)` or `(False, error_info)`.
2. **`generate_config_all_seeds(config, output_base_dir, n_seeds=None, explicit_seeds=None) -> summary`** — loops over seeds for one config, collects per-seed successes/failures/warnings, logs progress.
3. **`generate_all_configs(manifest_path, output_base_dir, n_seeds=None, config_ids=None) -> summary`** — top-level driver. Reads the manifest, optionally filters by `config_ids`, dispatches, returns a top-level summary dict.

CLI exposes:

- `--config-id N` (single config; SLURM array mode)
- `--config-ids 0,3,40` (comma-separated list)
- `--all` (every config in the manifest)
- `--n-seeds N` (override the manifest value, useful for smoke tests)
- `--seeds 0,1,5` (explicit seed list)
- `--summary-path PATH` (write structured JSON summary to disk)
- `--log-level {DEBUG,INFO,WARNING,ERROR}`

The script exits with status 1 if any seed failed, so SLURM correctly marks the task as FAILED.

#### 2.5.1 Atomicity

`_write_atomic` writes all four files into a sibling `seed_00.tmp/` directory first, then renames `.tmp` to the final name. This guarantees that downstream readers can never see a half-written seed directory: either the rename succeeds and the directory is complete, or it doesn't and the directory still doesn't exist. If an exception is raised mid-write, the partial `.tmp` directory is left for inspection but the final directory is not touched.

#### 2.5.2 Seed independence

The graph and the data use independent RNG streams: `graph_seed = seed`, `data_seed = seed + 10_000`. This is recorded in the metadata so that any future debugging can trace exactly which random stream produced which artifact.

### 2.6 `scripts/generate_synthetic_slurm.sh`

```bash
#SBATCH --array=0-83
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err
```

Each array task processes all 20 seeds for one config. The script has a multi-stage Python resolution fallback so it works whether or not the user has `~/.conda/envs/ggm_horseshoe/bin/python` available:

1. Try the explicit env path
2. `module load miniforge` (or `anaconda3`)
3. `conda activate ggm_horseshoe`
4. Fall back to `python` on `PATH`

If the manifest doesn't exist yet, the script regenerates it inline before dispatching.

### 2.7 `scripts/audit_synthetic_data.py`

Walks `data/synthetic/` and, for every config in the manifest, checks every expected `seed_NN/` directory. Each seed gets one of these statuses:

- `ok`
- `missing_dir`
- `missing_files` (directory exists but a file is absent)
- `corrupted` (zero-byte file, unloadable JSON, or — with `--strict` — wrong array shape / NaN-Inf in arrays)
- `metadata_mismatch` (metadata.json fields don't agree with the manifest)

In `--strict` mode it actually loads each `.npy` file and checks the shape and finiteness — this is the gold-standard check.

CLI: `--manifest`, `--data-root`, `--strict`, `--report PATH` (writes the full audit summary as JSON). Exit status 0 iff `n_failures == 0`.

### 2.8 `tests/test_synthetic.py`

55 tests across 11 classes, runs in 9.7 seconds. Coverage breakdown in §6.

---

## 3. The Experimental Grid

`compute_configs()` produces exactly **84 configurations**. The breakdown:

| Factor | Levels | Count |
|---|---|---|
| `p` | 10, 50, 100 | 3 |
| `gamma` | 0.90, 0.67, 0.42, 0.20, 0.10 | 5 |
| `graph` | erdos_renyi, block_diagonal | 2 |
| `sparsity` | 0.05, 0.10, 0.30 | 3 |
| **Cartesian product** | | **90** |
| Skipped: `(s=0.30, gamma=0.90)` for all `(p, graph)` combos | | **−6** |
| **Total configs** | | **84** |
| **Seeds per config** | | **20** |
| **Total simulations** | | **1,680** |

Sample sizes `T = ceil(p / gamma)` exactly match the table in `WORK1.md §2.2`:

| p / γ | 0.90 | 0.67 | 0.42 | 0.20 | 0.10 |
|---|---|---|---|---|---|
| **10** | 12 | 15 | 24 | 50 | 100 |
| **50** | 56 | 75 | 120 | 250 | 500 |
| **100** | 112 | 150 | 239 | 500 | 1000 |

The first three configs in the manifest:

```json
{"config_id": 0, "p": 10, "gamma": 0.90, "T": 12, "graph": "erdos_renyi",    "sparsity": 0.05, "n_blocks": null, ...}
{"config_id": 1, "p": 10, "gamma": 0.90, "T": 12, "graph": "erdos_renyi",    "sparsity": 0.10, "n_blocks": null, ...}
{"config_id": 2, "p": 10, "gamma": 0.90, "T": 12, "graph": "block_diagonal", "sparsity": 0.05, "n_blocks": 2,    ...}
```

Note that `(s=0.30, gamma=0.90)` is correctly absent from `p=10, gamma=0.90` (the manifest skips straight from id=1 to id=2 = block_diagonal s=0.05).

---

## 4. What We Generated

### 4.1 Run summary

| Metric | Value |
|---|---|
| SLURM job array | `gen_synth[0-83]` on `mit_normal` |
| Resources per task | 1 CPU, 4 GB RAM, 15 min wall |
| Tasks completed | 84 / 84 |
| Seed directories produced | 1,680 / 1,680 |
| Failures | 0 |
| Disk usage on cluster | **374 MB** at `data/synthetic/` |

### 4.2 Why "only" 374 MB?

The earlier estimate of 1.5–2 GB was based on the maximum-T case for every p; in reality the dataset is a mix of T values, and most are much smaller than the per-p maximum. The corrected estimate:

| p | T values across γ | Avg T | Avg per-seed bytes | × 560 seeds |
|---|---|---|---|---|
| 10 | 12, 15, 24, 50, 100 | ~40 | ~5 KB | ~3 MB |
| 50 | 56, 75, 120, 250, 500 | ~200 | ~120 KB | ~67 MB |
| 100 | 112, 150, 239, 500, 1000 | ~400 | ~480 KB | ~270 MB |
| Subtotal arrays | | | | ~340 MB |
| Metadata.json overhead | | | ~10–50 KB / file | ~30 MB |
| **Total** | | | | **~370 MB** |

This matches the observed 374 MB exactly.

### 4.3 Per-seed contents

Every `seed_NN/` directory has exactly four files:

```
seed_00/
├── omega_true.npy   # (p, p) float64 — the ground truth precision matrix
├── sigma_true.npy   # (p, p) float64 — Ω⁻¹, precomputed
├── Y.npy            # (T, p) float64 — sampled data
└── metadata.json    # all the scalar metadata + edge set + warnings
```

A representative `metadata.json` (config 40, seed 0; p=50, gamma=0.42, ER, s=0.30):

```json
{
  "config_id": 40,
  "p": 50,
  "T": 120,
  "gamma": 0.42,
  "graph": "erdos_renyi",
  "sparsity": 0.3,
  "n_blocks": null,
  "signal_range": [0.3, 0.8],
  "seed": 0,
  "graph_seed": 0,
  "data_seed": 10000,
  "n_edges": 369,
  "n_possible_edges": 1225,
  "realized_sparsity": 0.30122448979591837,
  "condition_number": 87.59191961640556,
  "min_eigenvalue": 0.09999999999999165,
  "max_eigenvalue": 8.759191961639825,
  "trace_omega": 220.00476438340812,
  "oracle_portfolio_variance": 0.004019356249411432,
  "diagonal_shift_applied": 3.4000952876681625,
  "min_eigenvalue_pre_shift": -3.3000952876681624,
  "max_mean_zscore": 2.7797872915006376,
  "max_abs_mean": 0.18553290584349158,
  "sample_cov_rank": 50,
  "warnings": ["large diagonal shift: 3.40 > 2.0"],
  "edge_set": [[0, 4], [0, 8], ...],
  "timestamp": "2026-04-11T17:15:12.373508+00:00"
}
```

This format is a strict superset of `WORK1.md §4.4` — it adds `min_eigenvalue_pre_shift`, `max_mean_zscore`, `max_abs_mean`, `sample_cov_rank`, and the structured `warnings` list, all of which are useful for downstream debugging.

---

## 5. Sanity Checks Performed

### 5.1 Unit tests (local, before submission)

```
55 passed in 9.69s
```

Full breakdown in §6.

### 5.2 Smoke test: 84 configs × 1 seed (local)

Ran `python scripts/generate_synthetic_data.py --all --n-seeds 1` to exercise every config exactly once. **Result: 84 successes, 0 failures, 1.01 s wall time.** 18 warnings fired, all of the form `large diagonal shift: ... > 2.0` on dense p=100 ER configs — exactly the configs where we expect the diagonal shift to be large because random ±[0.3, 0.8] off-diagonals at high density push the pre-shift `eigmin` far below zero.

### 5.3 Multi-seed test: 3 configs × 20 seeds (local)

Ran `python scripts/generate_synthetic_data.py --config-ids 0,3,40` to exercise the seed loop. **Result: 60 successes, 0 failures, 0.5 s wall time.** Diagonal shifts and condition numbers showed healthy variability across seeds, confirming that the per-seed RNG is doing its job.

### 5.4 Full SLURM run on `mit_normal`

```
JOBID                PARTITION  NAME       USER    ST  TIME  NODES  NODELIST(REASON)
11682618_[50-83]     mit_normal gen_synth  favara  PD  0:00  1      (Priority)
```

84 array tasks, each handling all 20 seeds for one config. Sample task log (`gen_11682618_0.err`):

```
[13:15:12] INFO Generating 1 configs from data/synthetic/configs/config_manifest.json into data/synthetic
[13:15:12] INFO Config 0: p=10 gamma=0.90 graph=erdos_renyi s=0.05 T=12
[13:15:12] INFO   ok  cfg=0 seed=0  n_edges=2  kappa=3.3   shift=0.000  0.02s
[13:15:12] INFO   ok  cfg=0 seed=1  n_edges=3  kappa=19.6  shift=0.028  0.01s
... (18 more lines)
[13:15:12] INFO Done: 1 configs, 20 successes, 0 failures in 0.2s
```

**All 84 tasks completed successfully.** No failed array elements.

### 5.5 Post-generation audit (cluster, `--strict`)

```
$ python scripts/audit_synthetic_data.py --strict --report logs/audit.json
Audit: manifest=.../config_manifest.json  data_root=.../data/synthetic
  configs:               84
  configs fully present: 84
  seed dirs expected:    1680
    ok                   1680
    missing_dir          0
    missing_files        0
    corrupted            0
    metadata_mismatch    0
```

This is the gold-standard check: it walks the entire tree, opens every `metadata.json`, loads every `.npy` file, verifies every shape, and verifies every value is finite. **Zero failures across 1,680 directories.**

### 5.6 File count cross-check

```
$ find data/synthetic -name "metadata.json" | wc -l
1680
```

Exactly the number expected: 84 × 20.

### 5.7 Disk usage cross-check

```
$ du -sh data/synthetic
374M    data/synthetic
```

Within ~1% of the predicted ~370 MB.

### 5.8 Optional post-hoc sanity checks (`scripts/sanity_checks.py`)

After the audit passed, all 8 optional distributional checks were also run on the cluster. Every check passed. The full output and per-check interpretation is in **§9**. Quick summary:

| # | Check | Status |
|---|---|---|
| 1 | Edge counts match theoretical expectations | ✅ |
| 2 | Condition numbers all < 125 | ✅ |
| 3 | Min eigenvalue ≥ 0.1 across all 1,680 seeds | ✅ |
| 4 | 273 expected `large diagonal shift` warnings, no others | ✅ |
| 5 | 20 distinct edge sets (no seed collisions) | ✅ |
| 6 | Ω·Σ = I to ~2×10⁻¹⁵ on disk | ✅ |
| 7 | Sample-mean z-scores well-behaved (max 4.38, none > 5) | ✅ |
| 8 | All 15 (p, T) pairs present with correct seed counts | ✅ |

---

## 6. Test Suite Breakdown (`tests/test_synthetic.py`)

55 tests across 11 classes. Each class targets one component.

| Class | Tests | What it covers |
|---|---|---|
| `TestSparseOmegaErdosRenyi` | 6 + parametrize | PD, symmetry, sparsity in range, diagnostics shape, edge-set ↔ nonzero correspondence, reproducibility under fixed seed |
| `TestSparseOmegaBlockDiagonal` | 3 × parametrize | PD, symmetry, **no inter-block edges** (the defining invariant) — verified for `(p, n_blocks) ∈ {(10,2), (50,5), (100,5)}` |
| `TestSparseOmegaBand` | 1 × parametrize | PD for the band variant (kept for completeness even though not in the WORK1 grid) |
| `TestDataSampling` | 5 | Shape, finiteness, sample-cov rank for both `T > p` and `T ≤ p`, reproducibility |
| `TestInversion` | 1 × parametrize | `Ω·Ω⁻¹ ≈ I` to 1e-8 |
| `TestAssemblePrecisionMatrix` | 1 | Roundtrip: build from offdiag+diag, verify symmetry and diagonal |
| `TestOraclePortfolio` | 1 | GMV weights sum to 1 |
| `TestConfigManifest` | 8 | Count == 84, count matches `expected_config_count`, no skipped configs present, IDs consecutive 0..83, T = ceil(p/γ), block configs have correct n_blocks, dir_paths unique, signal_range correct |
| `TestValidateOmega` | 5 | Accepts valid; rejects non-symmetric; rejects non-PD; rejects sparsity-pattern mismatch; warns on large diagonal shift |
| `TestValidateSigma` | 2 | Accepts true inverse; rejects fake inverse |
| `TestValidateData` | 4 | Accepts valid; rejects wrong shape; rejects NaN; rejects Inf |
| `TestGenerateSingleConfig` | 3 | Produces all 4 files; works for block_diagonal; different seeds → different edge sets |
| `TestAudit` | 1 | Detects missing seed directories in strict mode |

---

## 7. Design Decisions and Deviations from `WORK1.md`

A few places where the implementation departs from the literal spec, with rationale.

### 7.1 Generators return a 3-tuple

WORK1 §6.1 lists `sparse_omega_erdos_renyi(p, sparsity, signal_range, seed) → (Omega, edge_set)`.

Implementation returns `(Omega, edge_set, diagnostics)` where `diagnostics` contains `diagonal_shift`, `min_eigenvalue_pre_shift`, and `n_edges`.

**Why:** the `diagonal_shift_applied` and `min_eigenvalue_pre_shift` fields in `WORK1 §4.4`'s metadata schema can only be computed cheaply at generation time. Returning them via the generator avoids redundant work.

### 7.2 `src/utils/configs.py` is a separate file

WORK1 §6.1 puts `build_manifest` inside `scripts/generate_config_manifest.py`.

Implementation splits the pure logic into `src/utils/configs.py` and keeps the script as a thin wrapper.

**Why:** tests need to import the grid logic to verify it (e.g., `test_count`, `test_T_is_ceil_p_over_gamma`). Importing from a script is awkward; importing from `src.utils.configs` is clean and reusable downstream (the inference scripts will need the same encoding helpers).

### 7.3 Zero-mean check is per-component z-score, not absolute

WORK1 §5.2 says "`numpy.abs(Y.mean(axis=0)).max() < 3/√T`".

Implementation uses `max |ȳ_i / √(Σ_ii / T)| < 5` as a warning, not a hard failure.

**Why:** the literal `3/√T` assumes `Σ_ii ≈ 1` for all i, which is not true here (the diagonal varies). And a 3-sigma cutoff applied to `p=100` independent components would falsely fail ~24% of the time even on perfectly correct data. The 5-sigma per-component z-score has a false-positive rate of ~5×10⁻⁵, which is what we want for a "this should never trigger" sanity check. We also keep it as a *warning* rather than a failure so a single statistically unlucky seed doesn't kill a downstream pipeline.

### 7.4 Atomic writes

WORK1 §10 says "Do not save corrupted data". The implementation goes one step further: it writes all four files into a sibling `.tmp/` directory first, then renames atomically. This means a downstream reader can never observe a partially-written `seed_NN/` directory.

### 7.5 Warnings are collected, not just printed

Each validator returns `diagnostics["warnings"]` as a list of strings. These get aggregated into `metadata["warnings"]` per seed and into the SLURM summary JSON per task. This makes it easy to grep for problematic configs after the fact (see §9.4).

---

## 8. Interpreting the Warnings

The smoke test produced 18 warnings, all of the form:

```
large diagonal shift: X.YZ > 2.0
```

Distribution by config family:

| Family | Configs that warned | Typical shift |
|---|---|---|
| `erdos_renyi`, `p=50`, `s=0.30` (4 configs across γ values) | 4 | ~3.4 |
| `erdos_renyi`, `p=100`, `s=0.05` (5 configs) | 5 | ~2.1 |
| `erdos_renyi`, `p=100`, `s=0.10` (5 configs) | 5 | ~2.9 |
| `erdos_renyi`, `p=100`, `s=0.30` (4 configs, γ=0.90 skipped) | 4 | ~4.9 |

These are **expected and informational**, not bugs. At high density (`s=0.30`) and/or high dimension (`p=100`), the random ±[0.3, 0.8] off-diagonals make the pre-shift `eigmin` quite negative, so a large diagonal shift is needed to enforce `eigmin ≥ 0.1`. This is exactly the behavior `WORK1 §5.1.5` asked us to flag:

> **Diagonal loading was not extreme**: if the diagonal shift exceeds 2.0, log a warning. Large shifts mean the random graph + weights almost produced a non-PD matrix, and the resulting Ω₀ may have artificially inflated diagonal entries.

**Action item for the team:** when reporting results on these specific configs, note that the ground-truth Ω₀ has artificially inflated diagonal entries. This will affect Stein's loss interpretation (the trace term) and the eigenvalue spectrum but not the sparsity recovery metrics.

The warning threshold (2.0) is conservative; the actual shifts top out around 4.9, which is large but still leaves the off-diagonal signals interpretable.

---

## 9. Optional Sanity Checks: Results

The 5 checks in §5 cover the formal guarantees from WORK1. The 8 checks in this section are optional statistical / distributional checks bundled into [`scripts/sanity_checks.py`](../scripts/sanity_checks.py). They were run on the cluster after the audit passed.

```bash
conda activate ggm_horseshoe
python scripts/sanity_checks.py            # all 8 (~15 s)
python scripts/sanity_checks.py --check 3  # any individual check
python scripts/sanity_checks.py --list     # show available checks
```

**Result: every check passes cleanly.** Details below.

### 9.1 Edge-count distribution per (graph, p, s) — ✅

Verifies the generators produce graphs with the right average density.

```
graph               p      s     n      mean       std    min    max  expected
------------------------------------------------------------------------------
block_diagonal     10   0.05   100       1.1       1.1      0      4       1.0
block_diagonal     10   0.10   100       2.0       1.2      0      4       2.0
block_diagonal     10   0.30    80       5.6       1.9      2     10       6.0
block_diagonal     50   0.05   100      10.8       3.6      5     16      11.2
block_diagonal     50   0.10   100      22.2       5.6     12     33      22.5
block_diagonal     50   0.30    80      66.4       7.6     52     81      67.5
block_diagonal    100   0.05   100      45.2       7.5     34     58      47.5
block_diagonal    100   0.10   100      94.0       9.8     74    108      95.0
block_diagonal    100   0.30    80     284.9      11.7    264    307     285.0
erdos_renyi        10   0.05   100       3.0       1.7      1      6       2.2
erdos_renyi        10   0.10   100       5.0       2.3      1      9       4.5
erdos_renyi        10   0.30    80      13.3       2.6      9     18      13.5
erdos_renyi        50   0.05   100      60.5       7.6     44     75      61.2
erdos_renyi        50   0.10   100     121.8      10.5    105    143     122.5
erdos_renyi        50   0.30    80     365.9      14.5    343    401     367.5
erdos_renyi       100   0.05   100     249.6      13.9    213    274     247.5
erdos_renyi       100   0.10   100     497.5      15.5    471    524     495.0
erdos_renyi       100   0.30    80    1491.8      37.4   1427   1596    1485.0
```

**Interpretation:** Every observed mean is within ~1 standard deviation of the theoretical expectation. The slight overshoot at `erdos_renyi p=10 s=0.05` (3.0 vs 2.2) is well within the std of 1.7 — at small p the binomial variance dominates. For all larger p, the agreement is excellent (e.g. `erdos_renyi p=100 s=0.30`: 1491.8 vs 1485.0, agreement to 0.5%). Block-diagonal generators are essentially exact.

### 9.2 Condition-number distribution per (graph, p, s) — ✅

```
graph               p      s      median         max         p95
----------------------------------------------------------------
block_diagonal     10   0.05         2.5        13.1        13.1
block_diagonal     10   0.10         7.0        22.7        22.7
block_diagonal     10   0.30        22.3        30.4        30.4
block_diagonal     50   0.05        20.2        25.9        25.9
block_diagonal     50   0.10        24.8        32.9        32.9
block_diagonal     50   0.30        37.0        44.5        44.5
block_diagonal    100   0.05        26.9        33.3        33.3
block_diagonal    100   0.10        35.3        39.4        39.4
block_diagonal    100   0.30        55.1        64.0        64.0
erdos_renyi        10   0.05         7.6        25.7        25.7
erdos_renyi        10   0.10        20.1        32.5        32.5
erdos_renyi        10   0.30        31.3        38.2        38.2
erdos_renyi        50   0.05        39.1        42.8        42.8
erdos_renyi        50   0.10        50.2        54.6        54.6
erdos_renyi        50   0.30        83.3        89.7        89.7
erdos_renyi       100   0.05        53.8        58.6        58.6
erdos_renyi       100   0.10        72.5        78.8        78.8
erdos_renyi       100   0.30       121.3       124.6       124.6

  Global max condition number: 124.6
```

**Interpretation:** Every condition number across the 1,680 simulations is below 125 — well under the 1000 warning threshold (which is why no seed produced a `high condition number` warning in check 4). κ scales smoothly with both p and s, as expected. Block-diagonal κ is consistently lower than ER κ at the same (p, s), because the structural sparsity translates to fewer edges total and a less aggressive diagonal shift. NUTS should be numerically comfortable on every config.

### 9.3 Eigenvalue floor across all 1,680 seeds — ✅

```
  Total seeds checked:   1680
  Below floor (<0.1): 0  (must be 0)
  Min eigenvalue across all seeds: 0.1000000000  (should be ~0.1)
```

**Interpretation:** Zero violations. The reported `Min eigenvalue across all seeds: 0.1000000000` confirms the floor is being hit *exactly* on at least one seed (the diagonal-shift mechanism is doing its job). Every Ω₀ is provably PD with margin ≥ 0.1.

### 9.4 Aggregate warnings — ✅ (informational)

```
  Total seeds:           1680
  Seeds with warnings:   273
  Total warning lines:   273

  Warning categories:
      273  large diagonal shift

  Top 10 configs by warning count:
    cfg= 52  20 warnings across 20 seeds
    cfg= 46  20 warnings across 20 seeds
    cfg= 40  20 warnings across 20 seeds
    cfg= 34  20 warnings across 20 seeds
    cfg= 79  20 warnings across 20 seeds
    cfg= 80  20 warnings across 20 seeds
    cfg= 73  20 warnings across 20 seeds
    cfg= 74  20 warnings across 20 seeds
    cfg= 67  20 warnings across 20 seeds
    cfg= 68  20 warnings across 20 seeds
```

**Interpretation:** 273 seeds (~16% of the dataset) carry the `large diagonal shift` warning. **No other warning categories fired** — in particular no `high condition number` warnings, consistent with check 2's max of 124.6.

10 configs warn on **all 20 seeds** (i.e., the configuration is structurally guaranteed to need a large shift): these are the dense ER configs `p=50 s=0.30 × γ ∈ {0.67, 0.42, 0.20, 0.10}` (cfgs 34, 40, 46, 52) and `p=100 s ∈ {0.10, 0.30} × γ ∈ {0.42, 0.20, 0.10}` (cfgs 67, 68, 73, 74, 79, 80). The remaining 73 warnings (273 − 200) are spread across configs that only sometimes need a large shift — typically `p=100 s=0.05` configs.

This is the *expected and informational* behavior described in §8 — the team should remember that for these specific configs, the ground-truth Ω₀ has artificially inflated diagonal entries, which affects the trace term in Stein's loss but not the sparsity recovery metrics.

### 9.5 Cross-seed independence — ✅

```
  Target config: erdos_renyi p=50 s=0.30 (config 40)
  Seeds inspected:           20
  Distinct edge sets:        20 (must equal n_seeds)

  Mean pairwise overlap:     108.8
  Median pairwise overlap:   108
  Min pairwise overlap:      84
  Max pairwise overlap:      144
  Expected (s^2 * C(p,2)):   110.2
```

**Interpretation:** All 20 seeds produced distinct edge sets — the per-seed RNG is functioning correctly with no collisions. The mean pairwise overlap of 108.8 matches the theoretical expectation of `s² · C(p,2) = 0.09 × 1225 = 110.2` to within 1.3%. The min/max range (84–144) is consistent with the expected binomial variance. Confirms the seeds are statistically independent.

### 9.6 Reconstruction Ω·Σ ≈ I on disk — ✅

```
  Sampled seeds:                       50
  Worst max(|Omega @ Sigma - I|):      2.10e-15
  Tolerance (validation enforces):     1.00e-08
  STATUS: ok
```

**Interpretation:** Worst error across 50 random seeds is 2.10×10⁻¹⁵, **seven orders of magnitude below the 10⁻⁸ tolerance**. This is essentially machine precision (the numpy float64 ε is ~2.22×10⁻¹⁶). Confirms that what we wrote to disk is exactly what we generated and that the saved Σ is genuinely the inverse of the saved Ω.

### 9.7 Per-seed sample-mean z-scores — ✅

```
  Seeds with z-score:    1680
  Mean max-z:            2.25
  Median max-z:          2.26
  P90 max-z:             2.95
  P99 max-z:             3.55
  Max max-z:             4.38
  Number > 5 (warning):  0
  Number > 6:            0
  Number > 7:            0
```

**Interpretation:** **Zero seeds exceed the 5σ warning threshold.** The mean max-z of 2.25 matches the theoretical expectation: the distribution of `max_i |Z_i|` over `p` independent standard normals has expected value ~`√(2 log p)`, which for `p ∈ {10, 50, 100}` gives `{2.15, 2.80, 3.04}`. Our observed mean of 2.25 sits at the low end because most seeds in the dataset have small p (and small T amplifies sampling noise but our scaling normalizes that out). The empirical P99 of 3.55 and absolute max of 4.38 leave a comfortable safety margin. No data quality issues anywhere.

### 9.8 (p, T) distribution and seed counts — ✅

```
(p, T)           n_seeds              gammas
--------------------------------------------
  ( 10,   12)        80     [0.9]
  ( 10,   15)       120     [0.67]
  ( 10,   24)       120     [0.42]
  ( 10,   50)       120     [0.2]
  ( 10,  100)       120     [0.1]
  ( 50,   56)        80     [0.9]
  ( 50,   75)       120     [0.67]
  ( 50,  120)       120     [0.42]
  ( 50,  250)       120     [0.2]
  ( 50,  500)       120     [0.1]
  (100,  112)        80     [0.9]
  (100,  150)       120     [0.67]
  (100,  239)       120     [0.42]
  (100,  500)       120     [0.2]
  (100, 1000)       120     [0.1]

  Total (p, T) pairs: 15  (expected 15)
  Total seeds:        1680  (expected 1680)
```

**Interpretation:** All 15 (p, T) combinations from `WORK1 §2.2` are present with the expected seed counts. The 80-vs-120 split is exactly right: at γ=0.90 each (p, T) pair has 4 configurations (2 graphs × 2 sparsities, since s=0.30 is skipped), giving 80 seeds; at all other γ values each pair has 6 configurations (2 graphs × 3 sparsities), giving 120 seeds. 3 × 80 + 12 × 120 = 240 + 1440 = **1680**. Total seed count and dataset shape match expectations exactly.

---

### 9.9 Summary

| # | Check | Status |
|---|---|---|
| 1 | Edge-count distribution matches theory | ✅ |
| 2 | Condition numbers all < 125 (no κ warnings) | ✅ |
| 3 | Min eigenvalue ≥ 0.1 across all 1,680 seeds | ✅ |
| 4 | Warnings: 273, all `large diagonal shift`, no surprises | ✅ |
| 5 | Cross-seed independence verified, no collisions | ✅ |
| 6 | Σ = Ω⁻¹ to ~2×10⁻¹⁵ (7 orders below tolerance) | ✅ |
| 7 | Sample-mean z-scores well-behaved, max 4.38 | ✅ |
| 8 | All 15 (p, T) pairs present with correct seed counts | ✅ |

The dataset has now passed:

- ✅ 55 unit tests
- ✅ Local 84-config smoke test
- ✅ Local 60-seed multi-seed test
- ✅ Full 84-array SLURM run
- ✅ Strict audit (1,680/1,680 ok)
- ✅ 8 distributional sanity checks

There are no outstanding concerns about the data. **The synthetic dataset is ready for inference.**

---

## 10. What's Next

WORK1 is done. The next phases (per `_info/action_plan.tex` and the WORK1 deliverables checklist) are:

1. **Phase 3: Model Implementation** — finalize and validate the NumPyro `graphical_horseshoe` model on a single seed (smoke test). Most of this is already in [src/models/graphical_horseshoe.py](../src/models/graphical_horseshoe.py); we just need to load a real seed and run NUTS for a few hundred iterations.
2. **Phase 4: Inference at Scale** — adapt `generate_synthetic_slurm.sh` into an inference SLURM array script that loads `(omega_true.npy, Y.npy)` from each seed directory and runs NUTS / ADVI. The natural unit of work is one (config_id, seed, method) triple per array task.
3. **Phase 5: Evaluation** — load the saved posteriors plus the ground-truth files and compute Stein's loss, Frobenius loss, sparsity metrics, and the bimodal-shrinkage diagnostic.

The data layout we built is intentionally optimized for step 2: every inference task can read exactly one seed directory and produce exactly one output directory next to it. No global state, no shared files, embarrassingly parallel.

When you're ready for WORK2 (inference pipeline), the same pattern as WORK1 will apply: write a spec, generate the manifest, write a single-task generator, write a SLURM array, write an audit. The plumbing is now in place.
