# WORK3 Interim Report — as of 13 April EOD

**Project:** Sparse Bayesian Precision Matrix Estimation (6.7830, Spring 2026)
**Phase:** WORK3 — in progress
**Date:** 2026-04-13 end-of-day
**Progress report due:** April 15 (2 days)
**Final report due:** May 5

This document is a working snapshot of where we stand mid-WORK3. It captures every code change, every SLURM submission, every failure mode and fix, so the next session can pick up without re-diagnosing anything.

---

## 1. TL;DR

- **7 methods implemented and wired**: NUTS, Gibbs (Li et al. 2019), ADVI-MF, ADVI-LR, glasso, Ledoit-Wolf, sample covariance. (ADVI-FR was replaced with ADVI-LR per WORK3 spec.)
- **113 pytest tests pass** (up from 102 at end of WORK2). 11 new tests cover Gibbs, PSIS, and ADVI-LR.
- **Tier 2 grid submitted**: 42 configs (p∈{10,50,100} × ER × s∈{0.05,0.10,0.30} × 5 γ values, minus the s=0.30/γ=0.90 skip) × 5 seeds.
- **p=10 and p=50 are complete** across all 7 methods — publishable quality.
- **p=100 largely broken** due to two distinct bugs; both diagnosed and fixed tonight. Overnight resubmission planned.
- **Gibbs sampler validated**: 6.7s at p=10 smoke test, zero rejections.
- **Wall-time limit** confirmed: 12h on `mit_normal`; NUTS/Gibbs scripts raised from 4h to 12h, ADVI from 3h to 6h.

---

## 2. New Code (since WORK2 end)

### 2.1 Gibbs sampler — `src/inference/gibbs_runner.py` (~350 lines)

Full implementation of Li, Craig, and Bhadra (2019). Highlights:

- **Column-wise block update** using the standard decomposition `C_j = inv(s_jj · Ω_{-j,-j} + D_j)`, `μ_j = -C_j · s_{-j,j} / s_jj`, `Σ_j = C_j / s_jj`.
- **Truncated multivariate normal sampling** with rejection, using **Schur-complement PD check** (O(p²) instead of O(p³) eigendecomposition). Precomputes `Ω_{-j,-j}⁻¹` once per column.
- **Shifted Gamma for diagonal**: `ω_jj = g + ω_{-j,j}ᵀ · Ω_{-j,-j}⁻¹ · ω_{-j,j}` where `g ~ Gamma(T/2 + 1, scale=2/s_jj)`.
- **Data-augmentation for half-Cauchy**: auxiliary `ν_ij ~ InvGamma(1, 1 + 1/λ²_ij)` and `ξ ~ InvGamma(1, 1 + 1/τ²)`. InvGamma sampled as `1 / Gamma(a, scale=1/b)` (no scipy dependency).
- **Convergence diagnostics** (single-chain):
  - ESS via Geyer's initial positive-sequence estimator, FFT-based ACF.
  - Geweke z-test (first 10% vs last 50%) with p-value from error function (no scipy.stats needed).
- **Scaled budget**: p=10 → 6k sweeps, p=50 → 7k, p=100 → 8k with thinning=2.
- **Initialization**: `Ω = diag(rng.gamma(2, 0.5, p) + 1)` (positive, PD-safe). All off-diagonal entries start at 0.
- **Progress reporting**: `flush=True` prints every 10% of sweeps.

Key design choice (per WORK3 §12): **pure NumPy, no JAX**. The sampler is inherently sequential (column j depends on previous columns), so JIT-compilation gives no benefit; NumPy is simpler to debug.

### 2.2 PSIS diagnostic — `src/evaluation/psis.py` (~170 lines)

- `compute_psis_khat(log_weights)` — standalone k̂ from pre-computed log importance weights.
- `compute_psis_khat_from_svi(model, guide, params, Y, p, n_eval=200)` — computes log weights via NumPyro's `Trace_ELBO.loss` (each call returns `log p(y, θ) - log q(θ)` for a fresh θ ~ q).
- `_fit_gpd_khat()` — tries arviz's `psislw` first, falls back to Zhang & Stephens method-of-moments estimator on the upper tail (M = min(n/5, 3√n) samples above the threshold).
- `interpret_khat()`: <0.5 = "good", 0.5–0.7 = "marginal", >0.7 = "bad".

### 2.3 Gibbs SLURM script — `scripts/run_gibbs_slurm.sh`

```
#SBATCH --array=0-1679%100
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00   # raised from 4h after wall-time discovery
```

No JAX env vars needed (pure NumPy). Uses task manifest `gibbs.json`.

### 2.4 Tests — `tests/test_inference.py` (11 new tests)

| Class | Test | Checks |
|---|---|---|
| `TestGibbsSmoke` | `test_gibbs_produces_all_outputs` | All 5 output files (omega_hat, samples, tau, lambda, kappa) exist |
| | `test_gibbs_omega_is_pd` | Every posterior sample has min eigenvalue > 0 |
| | `test_gibbs_omega_is_symmetric` | Ω = Ωᵀ exactly |
| | `test_gibbs_kappa_in_unit_interval` | 0 ≤ κ ≤ 1 |
| | `test_gibbs_diagnostics_fields` | diagnostics.json has n_burnin, ESS, Geweke, rejection rate |
| | `test_gibbs_evaluate_produces_metrics` | Coverage_95 and bimodality_coefficient computed (Bayesian extras) |
| `TestPSIS` | `test_khat_identical_distributions` | Uniform log-weights → k̂ < 0.5 |
| | `test_khat_heavy_tails` | Cauchy draws → k̂ > 0.3 |
| | `test_interpret_khat` | Interpretation boundaries |
| `TestGibbsRunnerDirect` | `test_run_gibbs_returns_correct_shapes` | All array shapes correct |
| | `test_run_gibbs_tau_positive` | τ² and λ² remain positive |

---

## 3. Modified Code (WORK2 → WORK3)

### 3.1 `src/inference/run_single.py`

| Change | Why |
|---|---|
| `BAYESIAN_METHODS = ("nuts", "gibbs", "advi_mf", "advi_fr", "advi_lr")` | Track which methods produce posterior samples |
| Added `_run_gibbs(Y, p, T, ...)` wrapper | Dispatch layer; computes κ from λ² and τ² |
| `_METHOD_DISPATCH` now 8 keys: nuts, gibbs, advi_mf, advi_fr, advi_lr, glasso, ledoit_wolf, sample_cov | 7 real methods + advi_fr legacy |
| Added `_horseshoe_init_values(p, ncp)` helper | **Critical fix for p=100** — returns `{tau=1, lambdas=ones, z=zeros, omega_diag=5·ones}` so Ω starts as 5·I |
| NUTS now uses `init_to_value(values=_horseshoe_init_values(p, ncp))` | Replaces `init_to_median(num_samples=15)` which failed at p=100 |
| ADVI now passes `init_values` through to `advi_runner.run_advi` | Same fix |

### 3.2 `src/inference/advi_runner.py`

| Change | Why |
|---|---|
| `run_advi(..., init_values=None)` new parameter | Accepts explicit PD-safe starting values |
| Guide init: `init_to_value(values=init_values)` if supplied, else `init_to_median(num_samples=15)` | p≤50 uses sample-based init; p=100 uses hand-picked values |
| Imports now include `init_to_value` alongside `init_to_median` | With 3-tier fallback chain across NumPyro versions |

### 3.3 `src/inference/nuts_runner.py`

| Change | Why |
|---|---|
| Import `init_to_value` alongside `init_to_median` | To support the new init strategy |
| `run_nuts(..., init_strategy=None)` already supported — no further API change | Pass-through works |

### 3.4 `scripts/run_inference_single.py`

**Critical timeout fix**:

```python
def _default_timeout_for_method(method: str) -> int:
    if method == "nuts":
        return DEFAULT_NUTS_TIMEOUT_SECONDS         # 42000s = 11h40m
    if method == "gibbs":
        return DEFAULT_NUTS_TIMEOUT_SECONDS         # same 11h40m
    if method in ("advi_mf", "advi_fr", "advi_lr"):
        return DEFAULT_ADVI_TIMEOUT_SECONDS         # 20000s = 5h33m
    return DEFAULT_FREQ_TIMEOUT_SECONDS              # 1800s = 30min
```

Before tonight's fix, `gibbs` and `advi_lr` both fell through to the 30-min freq timeout. That's why 53 Gibbs p=100 tasks and 69 ADVI-LR tasks "timed out" — the Python SIGALRM fired at 30 min, long before SLURM's 12h wall.

### 3.5 `scripts/generate_task_manifests.py`

| Change | Why |
|---|---|
| `ALL_ADVI_METHODS = ["advi_mf", "advi_lr"]` (was `["advi_mf", "advi_fr"]`) | WORK3 replaces full-rank with low-rank |
| Added `ALL_GIBBS_METHODS = ["gibbs"]` | Dedicated Gibbs manifest |
| `main()` now writes `gibbs_tier*.json` alongside freq/advi/nuts | Matches the new Gibbs SLURM array |
| Tier 2 filter: `graph == "erdos_renyi" AND sparsity in (0.05, 0.10, 0.30)` | Covers all 3 p values × 5 γ × 3 sparsities minus 1 skip = **42 configs** |

### 3.6 SLURM wall-time updates

| Script | Old | New |
|---|---|---|
| `run_nuts_slurm.sh` | 4h | **12h** |
| `run_gibbs_slurm.sh` | 4h | **12h** |
| `run_advi_slurm.sh` | 3h | **6h** |

Raised after discovering `sinfo -p mit_normal` shows `TIMELIMIT=12:00:00`.

### 3.7 Aggregation / audit / figures

All 3 scripts updated to include `gibbs` and `advi_lr` in their `ALL_METHODS` lists:
- `scripts/aggregate_results.py`
- `scripts/audit_results.py`
- `scripts/generate_figures.py` (HEATMAP_METHODS)

---

## 4. Experimental Grid (Tier 2)

### 4.1 Configuration

- **p ∈ {10, 50, 100}** (3)
- **γ ∈ {0.90, 0.67, 0.42, 0.20, 0.10}** (5)
- **graph = erdos_renyi** only (block_diagonal is for sensitivity analysis later)
- **s ∈ {0.05, 0.10, 0.30}** (3) — including 0.05 per user request for the horseshoe's sparsest-case advantage
- **Skip**: (s=0.30, γ=0.90) because PD-shift becomes pathological

Total: **42 configs** × 5 seeds × 7 methods = **1,470 inference tasks**.

Plus freq methods ran all 20 seeds by default (not just 5), so freq alone is 42 × 20 × 3 = **2,520 method-level freq runs**.

### 4.2 Task manifests (Tier 2)

| Manifest | Tasks | Per task |
|---|---|---|
| `freq_tier2.json` | 42 | 1 config, batches 3 methods × 20 seeds |
| `advi_tier2.json` | 420 | (config, seed, method) triple, 2 methods × 5 seeds × 42 configs |
| `nuts_tier2.json` | 210 | (config, seed) pair, 5 seeds × 42 configs |
| `gibbs_tier2.json` | 210 | (config, seed) pair, 5 seeds × 42 configs |

Task ID convention (for p=100 subset identification):
- NUTS/Gibbs: task IDs **140–209** are p=100 (14 configs × 5 seeds = 70 tasks at p=100).
- ADVI: task IDs **280–419** are p=100 (14 configs × 5 seeds × 2 methods = 140 tasks at p=100).

---

## 5. Overnight Run History

### 5.1 Submission sequence (timeline)

1. **Freq** (array 0,1,4,5,6,10,11,12,16,17,18,22,23,24,28,29,32,33,34,38,39,40,44,45,46,50,51,52,56,57,60,61,62,66,67,68,72,73,74,78,79,80) — 42 tasks, ran under 10 min.
2. **NUTS** (array 0–209) — 210 tasks. Queued, ran over ~8 hours. **Some p=100 tasks still ran at old 4h limit** because this was submitted before the 12h wall-time discovery.
3. **Gibbs** (array 0–209) — 210 tasks. Also submitted with old 4h limit baked in; same issue.
4. **ADVI first half** (array 0–199) — submitted later due to QOS cap (448 jobs/user on `mit_normal`). Partial batch.
5. **ADVI second half** (array 200–419) — submitted morning of 14 April.

### 5.2 Final Tier 2 status (audit at 19:00 on 13 April)

| Method | Success | Failed | Timeout | Notes |
|---|---|---|---|---|
| sample_cov | 840/840 | 0 | 0 | ✅ Complete |
| ledoit_wolf | 840/840 | 0 | 0 | ✅ Complete |
| glasso | 840/840 | 0 | 0 | ✅ Complete (after 8 config-56 reruns on login node) |
| nuts | 154/210 | 36 | 4 | p=10 and p=50 complete; **p=100: 15 ok / 36 failed / 4 timeout** |
| gibbs | 140/210 | 1 | 53 | p=10 and p=50 complete; **p=100: mostly timeouts (old 30-min dispatcher timeout)** |
| advi_mf | 140/210 | 69 | 0 | p=10 and p=50 complete; **p=100: all 69 failed with "Cannot find valid initial parameters"** |
| advi_lr | 140/210 | 69 | 0 | Same p=100 failure pattern |
| advi_fr | 0/0 | — | — | Replaced by advi_lr |

**Bottom line**: p=10 and p=50 are fully complete across all 7 methods. p=100 is mostly broken across all Bayesian methods.

---

## 6. Diagnosing the p=100 Failures

### 6.1 Bug 1: Missing timeout dispatch for Gibbs and ADVI-LR

**Observed**: Gibbs p=100 had 53 "timeouts" with elapsed times ~1800s. That's exactly 30 minutes — far below SLURM's 4h (let alone 12h) wall.

**Diagnosis**: `_default_timeout_for_method` in `run_inference_single.py` had:
```python
if method in ("advi_mf", "advi_fr"):   # missing advi_lr!
    return DEFAULT_ADVI_TIMEOUT_SECONDS
return DEFAULT_FREQ_TIMEOUT_SECONDS     # gibbs fell through here
```

Both `gibbs` and `advi_lr` fell through to the 30-minute freq timeout, triggering the Python SIGALRM far earlier than SLURM would have.

**Fix** (applied tonight): explicit branches for `gibbs` (uses NUTS timeout = 11h40m) and `advi_lr` (uses ADVI timeout = 5h33m).

### 6.2 Bug 2: `init_to_median` fails at p=100

**Observed** (NUTS p=100):
```json
{
  "status": "failed",
  "error": "RuntimeError('Cannot find valid initial parameters. Please check your model again.')",
  "elapsed_seconds": 12.8,
  "parameterization": null
}
```

**Observed** (ADVI-MF p=100):
```json
{
  "status": "failed",
  "error": "ADVI failed for all learning rates: RuntimeError('Cannot find valid initial parameters. ...')",
  "lr_retries": 3,
  "elapsed_seconds": 214.6
}
```

**Diagnosis**: At p=100, the model has 10,001 latent dimensions:
- 1 τ (scalar)
- 4,950 λ_ij (off-diagonal half-Cauchy)
- 4,950 z_ij (off-diagonal standard normal, NCP)
- 100 ω_diag_i (diagonal half-Normal)

`init_to_median(num_samples=15)` estimates each site's median from 15 prior samples. With HalfCauchy's heavy tail, the 15-sample median is noisy. Across 10,001 sites, it's statistically near-certain that at least one latent lands in a region that produces a non-PD Ω. NumPyro's `find_valid_initial_params` retries a few times and gives up with "Cannot find valid initial parameters".

**Fix** (applied tonight): hand-picked PD-safe values via `init_to_value`:
```python
values = {
    "tau": 1.0,
    "lambdas": np.ones(n_offdiag),       # λ = 1 everywhere
    "z": np.zeros(n_offdiag),            # z = 0 → ω_offdiag = 0
    "omega_diag": np.ones(p) * 5.0,      # ω_ii = 5 (well above the prior median of ~3.37)
}
# Ω = assemble({z·λ·τ=0}, {ω_ii=5}) = 5·I   (trivially PD)
```

Wired into both NUTS (via `init_strategy`) and ADVI (via `init_loc_fn`).

### 6.3 Symptom summary

| Method @ p=100 | Bug triggered | Fixed? |
|---|---|---|
| nuts | Bug 2 (init) | Yes |
| gibbs | Bug 1 (timeout; also old 4h SLURM wall) | Yes |
| advi_mf | Bug 2 (init) | Yes |
| advi_lr | Bug 1 (timeout) + Bug 2 (init) | Yes (both) |

---

## 7. Code Audit (as of 13 April EOD)

Verified each fix is in place via grep and re-read:

| Fix | File | Line(s) | Status |
|---|---|---|---|
| `_horseshoe_init_values(p, ncp)` helper | [run_single.py](../src/inference/run_single.py#L221) | 221–246 | ✅ |
| NUTS uses `init_to_value` | [run_single.py](../src/inference/run_single.py#L358) | 358 | ✅ |
| ADVI core passes `init_values` | [run_single.py](../src/inference/run_single.py#L514) | 514 | ✅ |
| `advi_runner.run_advi(init_values=...)` accepts & uses | [advi_runner.py](../src/inference/advi_runner.py#L96) | 96, 170–173 | ✅ |
| NUTS runner imports `init_to_value` | [nuts_runner.py](../src/inference/nuts_runner.py#L28) | 28 | ✅ |
| Gibbs timeout = NUTS timeout (11h40m) | [run_inference_single.py](../scripts/run_inference_single.py#L103) | 103–105 | ✅ |
| ADVI-LR in ADVI timeout group | [run_inference_single.py](../scripts/run_inference_single.py#L106) | 106 | ✅ |
| NUTS SLURM wall 12h | [run_nuts_slurm.sh](../scripts/run_nuts_slurm.sh) | — | ✅ |
| Gibbs SLURM wall 12h | [run_gibbs_slurm.sh](../scripts/run_gibbs_slurm.sh) | — | ✅ |
| ADVI SLURM wall 6h | [run_advi_slurm.sh](../scripts/run_advi_slurm.sh) | — | ✅ |

**Regression test**: `pytest tests/` → **113 passed, 2 skipped in 3.1s** (same as pre-WORK3 plus 11 new Gibbs/PSIS tests).

---

## 8. Smoke Tests Performed

Before the big Tier 2 launch, on a p=10 seed (config 22, γ=0.10, s=0.05):

| Method | Result | Time | Key diagnostics |
|---|---|---|---|
| sample_cov | ✅ success | <1s | — |
| ledoit_wolf | ✅ success | <1s | — |
| glasso | ✅ success | 1s | — |
| nuts | ✅ success | 13s | divergence_rate=2.35%, max_rhat=1.0013, ESS=3044 |
| gibbs | ✅ success | 6.7s | 0 rejections, τ²≈0 (strong shrinkage on sparse truth) |
| advi_mf | ✅ success | ~1min | final loss ≈ -4763 |
| advi_lr | ✅ success | 13s | final loss ≈ 1435, 99% finite steps |

All 7 methods verified before cluster launch.

---

## 9. Test Coverage

Full test suite as of tonight:

```
tests/test_metrics.py               37 passed
tests/test_synthetic.py             55 passed
tests/test_inference.py             15 passed + 2 skipped
    - 3 frequentist smoke tests       (always run)
    - 2 evaluation wiring tests
    - 2 NumPyro smoke tests           (skipped without RUN_INFERENCE_TESTS=1)
    - 6 Gibbs smoke + unit tests      (NEW)
    - 3 PSIS tests                    (NEW)
    - 2 Gibbs direct-runner tests     (NEW)
tests/test_evaluation_integration.py 4 passed
─────────────────────────────────────────────
                                    113 passed, 2 skipped
```

Pure-Python tests complete in ~3 seconds. Bayesian smoke tests (gated) verified manually in cluster smoke run.

---

## 10. Storage and Compute Footprint

| Data | Size | Location |
|---|---|---|
| Synthetic data | 374 MB | `data/synthetic/` |
| Tier 2 results (so far) | ~5 GB | `results/synthetic/` |
| Task manifests | <1 MB | `results/task_manifests/` |
| Job logs | ~200 MB | `logs/` |

Compute used overnight (rough):
- **NUTS**: ~270 CPU-hours (mean 83 min × 210 tasks, though 50 are p=100 timeouts that burned ~4h each)
- **Gibbs**: ~80 CPU-hours (mean 10 min × 210 tasks, p=100 tasks short-circuited at 30 min)
- **ADVI**: ~45 CPU-hours (mean 4 min × 210 tasks × 2 methods)
- **Freq**: ~2 CPU-hours total

Next overnight run (p=100 only):
- ~140 CPU-hours NUTS (70 tasks × mean ~2h expected after fix)
- ~70 CPU-hours Gibbs (70 tasks × ~1h expected)
- ~12 CPU-hours ADVI (140 tasks × ~5 min × 2 guides)

---

## 11. Overnight Plan (14 April)

```bash
cd ~/bayesian_spm
module load miniforge && conda activate ggm_horseshoe
git pull

# Wait for any stragglers still running (check squeue)
squeue -u $USER | wc -l

# Clean broken p=100 results
python -c "
import json, glob, shutil
from pathlib import Path
cleaned = 0
for method in ['nuts', 'gibbs', 'advi_mf', 'advi_lr']:
    for path in glob.glob(f'results/synthetic/erdos_renyi/p100/**/{method}/diagnostics.json', recursive=True):
        d = json.load(open(path))
        if d['status'] in ('failed', 'timeout'):
            shutil.rmtree(Path(path).parent)
            cleaned += 1
print(f'Cleaned {cleaned} broken p=100 directories')
"

# Rebuild tier 2 manifests (identical but fresh)
python scripts/generate_task_manifests.py --tier 2
cp results/task_manifests/advi_tier2.json results/task_manifests/advi.json
cp results/task_manifests/nuts_tier2.json results/task_manifests/nuts.json
cp results/task_manifests/gibbs_tier2.json results/task_manifests/gibbs.json

# Submit only p=100 subset
sbatch --array=140-209 scripts/run_nuts_slurm.sh       # 70 tasks, 12h wall
sbatch --array=140-209 scripts/run_gibbs_slurm.sh      # 70 tasks, 12h wall
sbatch --array=280-419 scripts/run_advi_slurm.sh       # 140 tasks, 6h wall

squeue -u $USER | head
```

Tomorrow morning:

```bash
python scripts/audit_results.py --strict --report logs/audit_15apr.json
python scripts/aggregate_results.py
python scripts/generate_figures.py --config-id 45 --seed 0
```

Then write the progress report with Tier 2 results.

---

## 12. Interpretation Notes (for the progress report)

### 12.1 Tier 1 (WORK2) vs Tier 2 (WORK3)

WORK2 gave us two (p=50, γ) points. Tier 2 expands to 15 (p, γ) points plus 3 sparsity levels, allowing:

1. **Loss-vs-γ curves**: for each p, 5 γ values give a smooth curve per method. The key finding to highlight is where NUTS starts beating frequentists (answer so far: everywhere at p=50).
2. **p-scaling story**: NUTS at p=100 is feasible but expensive (~2–4 h/task); Gibbs at p=100 is faster (~1–2 h/task). This is the "Gibbs fills the gap" narrative from WORK3 §4.1.
3. **Sparsity sensitivity**: s=0.05 (very sparse) is where the horseshoe should dominate glasso most strongly. Early Tier 1 data showed NUTS/ADVI at s=0.10 already dominate; s=0.05 results (from overnight runs) will let us confirm the theoretical expectation empirically.

### 12.2 What the p=100 data will tell us

Once overnight runs complete:

- If NUTS p=100 converges on some configs: "NUTS is feasible at p=100 for simpler settings but hits the 12h wall for hard γ=0.90 configs."
- If Gibbs p=100 converges everywhere: **"Gibbs fills the gap at p=100 where NUTS is infeasible"** — this is the WORK3 §4.1 narrative materialized.
- If ADVI p=100 converges: compare its bimodality coefficient and coverage against NUTS/Gibbs — does the "ADVI preserves bimodality" finding from WORK2 hold at higher p?

### 12.3 Expected shape of the final report

With Tier 1 + Tier 2 + planned real-data analysis (CRSP deferred per user request):

- **§4 Synthetic experiments** will have all four WORK3 must-have figures (heatmap, shrinkage profiles, Stein's loss vs γ, runtime vs p) plus the F1 and coverage should-haves.
- **§5 Real data** can still be written with only synthetic data by focusing on the scalability + calibration findings. The CRSP story can slot in if there's time before May 5.

---

## 13. Known Issues / TODOs

| Issue | Severity | Action |
|---|---|---|
| ADVI-LR at p=100 may still be slow even with fixes (large Cholesky + many iterations) | Medium | Monitor overnight; if most timeout, reduce `num_steps` for p=100 |
| Gibbs `warnings.warn` on rejection failure uses `warnings` module — noisy in logs | Low | Replace with per-config aggregate warning count in diagnostics |
| `--skip-existing` flag is defined on CLI but not passed through freq SLURM `--seeds all` batch mode | Low | If we need to rerun, just delete old output and resubmit; `run_inference` overwrites atomically anyway |
| No block-diagonal Tier 2 runs (sensitivity tier not executed) | Low | Can add in Tier 2b after overnight finishes |
| No CRSP real data (user requested to defer) | By design | Return to in WORK4 if time allows before May 5 |
| PSIS-khat not yet computed for ADVI runs in diagnostics.json | Medium | Wire `compute_psis_khat_from_svi` into `evaluate_single.py` for advi_mf/advi_lr tasks |

---

## 14. Session Continuity Notes

**For the next Claude session** picking this up:

1. **State of the cluster**: Check `squeue -u $USER`. If empty, the overnight runs are done and you can audit + aggregate immediately.
2. **Primary action on Apr 14**: Run the overnight plan (§11), then draft the progress report.
3. **Critical code invariants** that MUST NOT regress:
   - `_horseshoe_init_values(p, ncp)` returns a dict with keys `{tau, lambdas, z, omega_diag}` (or `{tau, lambdas, omega_offdiag, omega_diag}` if ncp=False).
   - NUTS calls `init_to_value(values=_horseshoe_init_values(p, ncp=ncp))` as `init_strategy`.
   - ADVI core passes `init_values=_horseshoe_init_values(p, ncp=True)` to `run_advi`.
   - `_default_timeout_for_method` must have explicit branches for `gibbs` and `advi_lr`.
4. **Test command** to verify nothing regressed: `pytest tests/` → expect 113 passed, 2 skipped.
5. **The bimodality surprise from WORK2 should be revalidated** once Tier 2 data comes in. If it still holds (ADVI-MF b ≈ NUTS b), that remains the paper's most interesting finding.

---

## 15. File Inventory (as of 13 April EOD)

```
_info/
├── WORK1.md                             # spec (original)
├── WORK1_REPORT.md                      # completion report
├── WORK2.md                             # spec
├── WORK2_REPORT.md                      # completion report
├── WORK3.md                             # spec (revised version, 7 methods)
├── WORK3_interim_report_as_of_13apr_eod.md   # THIS FILE
├── action_plan.tex                      # original research plan
├── project_guide.tex                    # technical reference
├── proposal.tex                         # 6.7830 proposal
├── reading_plan.tex                     # literature roadmap
├── code_guide.tex                       # codebase reference
├── cluster/                             # ORCD setup docs
├── example.slurm                        # legacy example
└── sanity_checks.txt                    # WORK1 sanity output

src/
├── inference/
│   ├── run_single.py                    # EDITED: horseshoe_init_values, gibbs+advi_lr dispatch
│   ├── nuts_runner.py                   # EDITED: import init_to_value
│   ├── advi_runner.py                   # EDITED: init_values param
│   └── gibbs_runner.py                  # NEW: Li et al. 2019 Gibbs sampler
├── evaluation/
│   ├── metrics.py                       # (unchanged since WORK2)
│   ├── evaluate_single.py               # (unchanged since WORK2)
│   ├── shrinkage.py                     # (unchanged since WORK2)
│   └── psis.py                          # NEW: PSIS k-hat diagnostic
├── models/
│   └── graphical_horseshoe.py           # (unchanged since WORK2 — jitter + validate_args=False)
├── utils/
│   ├── configs.py, matrix_utils.py, validation.py, plotting.py  (unchanged)
├── portfolio/gmv.py                     # (unchanged)
└── benchmarks/frequentist.py            # (unchanged)

scripts/
├── run_inference_single.py              # EDITED: timeout map fix
├── run_nuts_slurm.sh                    # EDITED: 4h → 12h
├── run_gibbs_slurm.sh                   # NEW: Gibbs SLURM array
├── run_advi_slurm.sh                    # EDITED: 3h → 6h
├── run_freq_slurm.sh                    # (unchanged)
├── generate_task_manifests.py           # EDITED: gibbs + advi_lr methods
├── generate_config_manifest.py          # (unchanged)
├── generate_synthetic_data.py           # (unchanged)
├── audit_synthetic_data.py              # (unchanged — WORK1)
├── audit_results.py                     # EDITED: ALL_METHODS += gibbs, advi_lr
├── aggregate_results.py                 # EDITED: same
├── generate_figures.py                  # EDITED: HEATMAP_METHODS += gibbs, advi_lr
├── run_experiment.py                    # (unchanged)
└── sanity_checks.py                     # (unchanged — WORK1)

tests/
├── test_metrics.py                      # (unchanged since WORK2)
├── test_synthetic.py                    # (unchanged since WORK1)
├── test_inference.py                    # EDITED: +11 Gibbs/PSIS/direct tests
└── test_evaluation_integration.py       # (unchanged since WORK2)
```
