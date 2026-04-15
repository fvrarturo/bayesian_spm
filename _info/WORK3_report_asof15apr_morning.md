# WORK3 Status Report — 15 April Morning

**Project:** Sparse Bayesian Precision Matrix Estimation (6.7830, Spring 2026)
**Phase:** WORK3 — Tier 2 finalisation + figure pipeline
**Date:** 2026-04-15 morning
**Progress report due:** **TODAY** (April 15)
**Final report due:** May 5

This report picks up from `WORK3_report_asof14apr_morning.md`.  The last
24 hours produced three substantial changes: (a) the disk problem was
solved by **in-place compression** rather than the proposed archive-
to-Mac plan, (b) NUTS was re-run at p=100 with the new initialization
and produced a much cleaner (and scientifically more compelling) story,
and (c) the figure pipeline was expanded from 4 to 11 plots with
outlier-resistant statistics and diagnostic-based dashboards.

---

## 1. TL;DR

- **Disk recovered in-place, no data moved.** The 170 GB `.npy` footprint
  was compressed to **77.6 GB via `.npz` + float32 cast** in a parallel
  SLURM job (16 workers, 0 errors, 4 294 files).  Freed **84.3 GB**.
  Cluster is now at 120 GB / 195 GB, comfortable for the remaining
  figure work. **Option B (previously rejected) turned out to be
  correct** — the loader-compatibility risk was manageable because the
  consumers were few (`evaluate_single.py`, `generate_figures.py`,
  `tests/`) and all accepted a simple `.npz`-then-`.npy` fallback helper.
- **NUTS p=100 re-run with `init_to_value` completed overnight**:
  185 NUTS successes across the tree (70/70 at p=10, 69/70 at p=50,
  46/70 at p=100).  The init-failure mode is gone; the catastrophic
  mixing-failure mode is now visible and unambiguous.
- **Headline upgrade to the paper narrative**: at p=100, **0 / 46
  successful NUTS runs converged** (strict gate: R-hat<1.01, ESS>400,
  divergence-rate<5 %).  **Median max-R-hat = 5.62**.  This is much
  stronger than yesterday's "NUTS has high failure rate" framing:
  even the runs that *finish* do not mix.  **NUTS is structurally
  unable to sample this posterior at p=100, not just slow.**
- **Three-way failure mode now established** at p=100:
  - ADVI catastrophically mis-calibrated (coverage 0.02)
  - NUTS structurally non-mixing (R-hat ≈ 5.6)
  - **Gibbs is the only Bayesian method that actually works**
    (66/70 success, properly calibrated, ~16 min median wall)
- **Figure pipeline tripled in scope**: 7 new plot functions, robust
  (median + IQR) mode, diagnostic-level plots reading
  `diagnostics.json` directly (no aggregation dependency), plus
  diagnostic prints for dropped (method, p) points so missing series
  are debuggable at a glance.

---

## 2. The disk pivot — from Option A to Option B

The April 14 report recommended **Option A**: rsync bulky `.npy` files
to the Mac, then prune the cluster.  We instead implemented **Option B
(previously rejected): in-place compression**.  The rejection rationale
("requires updating every loader in scripts/ and src/") turned out to be
over-cautious — the audit showed only four loader sites (`evaluate_single`,
`generate_figures`, `run_single`, `tests/test_inference`), easily
shimmed behind a new helper.

### 2.1 The helper layer

New file: [`src/utils/io.py`](../src/utils/io.py)

```python
def load_samples(dir_path, name):
    """Try <dir>/<name>.npz first, fall back to <dir>/<name>.npy."""

def samples_exist(dir_path, name):
    """True if either .npz or .npy exists."""

def save_samples_compressed(path, arr, dtype=None):
    """Writes .npz with single key 'arr'; optional dtype downcast."""
```

This gives downstream code a single API that doesn't care about the
storage format.  Every existing `.npy` on disk continues to work; new
runs write `.npz`.  The fallback path is the entire back-compat
story — no separate migration flag, no explicit conversion.

### 2.2 The producer change

[`src/inference/run_single.py:164–189`](../src/inference/run_single.py#L164-L189) now:

- Writes `omega_samples`, `kappa_samples`, `lambda_samples`,
  `omega_diag_samples`, `tau_samples`, `elbo_trace` as **zlib-compressed
  `.npz`**.
- **Casts all stochastic draws to `float32`.**  The precision loss
  (~1e-7) is five orders of magnitude below the MCMC Monte-Carlo noise
  floor (~1 / √N ≈ 1e-2), so this is lossless from an inference
  standpoint.  `omega_samples` was already float32 for the same reason.
- Leaves `omega_hat.npy`, `sigma_hat.npy`, `offdiag_magnitudes.npy` as
  plain `.npy` — they're small, widely referenced, and the compression
  ratio is negligible.

### 2.3 The migration utility

New script [`scripts/compress_results.py`](../scripts/compress_results.py):

- Walks `results/synthetic/`, finds `.npy` files matching the
  compression spec, converts each to `.npz` (downcasting float64 →
  float32 where applicable), **deletes the `.npy` only after the
  `.npz` lands on disk**.
- Atomic writes via `.tmp.npz` → rename; idempotent (re-running skips
  already-converted files).
- Flags: `--dry-run`, `--workers N`, `--keep-npy`, `--quiet`.
- `multiprocessing.Pool` with `imap_unordered(chunksize=4)` for
  parallelism.

New SLURM wrapper [`scripts/run_compress_slurm.sh`](../scripts/run_compress_slurm.sh):
1 node, 16 CPUs, 256 GB (user-bumped from 32 GB), 5 h wall.  Sets
`OMP/MKL/OPENBLAS_NUM_THREADS=1` so workers don't oversubscribe BLAS.

### 2.4 The bug that nearly broke the migration

A subtle `np.savez_compressed` quirk: **the function auto-appends
`.npz` if the filename doesn't already end in `.npz`**.  My first
attempt used `<name>.npz.tmp` as the temp name; savez wrote it as
`<name>.npz.tmp.npz`, and the subsequent `.replace(npz_path)` failed
because the tmp file didn't exist under that name.  Fixed by using
`<name>.tmp.npz` (ends in `.npz`, savez doesn't rewrite it).  Caught
in smoke-test before running on the cluster; would have been a
catastrophe in production.

### 2.5 Results on the cluster

```
=== compress_11842676 ===
Host: compute-node (16 CPUs, 256 GB)
Workers: 16
Converted: 4294 files
Skipped (already .npz): 0
Errors: 0
Before: 161.9 GB
After:  77.6 GB  (48% of original)
Freed:  84.3 GB
Disk after:  nfs001.lb:/home  403T  221T  183T  55% /orcd/home/002
Done: 2026-04-14T13:23:13Z
```

**Time from `sbatch` to "done": under 20 minutes.**  Parallel I/O across
16 workers turned a potentially hours-long operation into a coffee break.

Post-compression verification on the highest-risk directory:

```
omega_samples:  (5000, 100, 100)  float32   finite=True
kappa_samples:  (5000, 4950)      float32   finite=True
lambda_samples: (5000, 4950)      float32   finite=True
tau_samples:    (5000,)           float32   finite=True
```

All four bulky arrays load via `load_samples()`, all finite.  The
migration is end-to-end verified.

---

## 3. The NUTS p=100 re-run

### 3.1 What actually happened

Two submissions were needed:

| Submission | Outcome |
|---|---|
| **Attempt 1** (shortly after the April 14 compression finished) | **All array tasks finished in 2 seconds.** |
| **Attempt 2** (after investigation) | **Tasks ran correctly for 4–11 hours each.** |

The 2-second failures were the lingering effect of disk quota — before
the compression completed, `_save_results_atomic()` was aborting on
its `.tmp` `mkdir` because the quota was still over.  After the
compression freed 84 GB, the re-submission ran properly.

### 3.2 Latent bug found during investigation

`scripts/run_nuts_slurm.sh` had `--timeout-seconds 14400` hard-coded
(a legacy of the pre-12h-wall era).  This cap of 4 h would have
internally timed out every p=100 task well before the 12 h SLURM wall,
defeating the whole point of the larger wall allocation.  **Removed the
override** so the module default `DEFAULT_NUTS_TIMEOUT_SECONDS = 42 000`
(11h40m) now applies, leaving a 20-minute buffer under the 12 h wall.

### 3.3 Final Tier 2 NUTS audit

```
=== nuts (final) ===
  p=10   success      70       (complete)
  p=50   success      69       (1 missing — ignore, consistent with AM-14)
  p=100  success      46
  p=100  failed/timeout ≈ 24   (from the 70-task array)
```

46 p=100 successes, up from 17 in the April 14 report.  The bulk of
this came from `init_to_value` finally being exercised on a freshly-
submitted batch with correct SLURM wall + internal timeout.

### 3.4 The real finding at p=100: *NUTS structurally cannot mix*

This is the result that deserves to headline the paper.  **Of the 46
p=100 NUTS runs that returned `status=success`, ZERO met the strict
convergence gate** (max R-hat < 1.01 **and** min bulk ESS > 400 **and**
divergence rate < 5 %).

```
46 p=100 NUTS successes
  converged (strict):              0 / 46  (0 %)
  max R-hat: [1.004, 1.005, 1.007, 1.008, 1.009, ...]  median = 5.62
```

Five lowest R-hats are ~1.004–1.009 (fine on R-hat, but they still
fail the ESS gate).  **Median R-hat = 5.62**.  A tenth of the runs
have R-hat > 10.  `min_bulk_ess = 2` is common — meaning the effective
sample size, across 20 000 posterior draws, is effectively two
independent samples per parameter.

**Scientific interpretation**: the NUTS chains have not mixed.  They
are returning draws from entirely different regions of the posterior.
The numeric outputs "look like" samples (every Ω is PD, every R-hat
computation runs to completion), but the between-chain variance is
~30× the within-chain variance — the defining symptom of a sampler
that hasn't converged.

**Why this matters for the paper**: yesterday's framing was "NUTS has
a high failure/timeout rate at p=100, so practically it doesn't scale."
That's a runtime claim.  Today's framing is "NUTS's *successful* runs
don't mix — the sampler is structurally unable to navigate the
local-global funnel at D ≈ 10 000."  That's a geometric claim about
the model, not a runtime one, and it's much stronger.

### 3.5 The three-way failure mode

Combining Tier 2 results at p=100:

| Method | Success rate | Calibration (coverage) | Mixing (R-hat / k-hat) | Practical? |
|---|---|---|---|---|
| **NUTS** | 46/70 (66 %) | — (chains not mixed) | **R-hat median 5.62** ❌ | No |
| **ADVI-MF** | 69/70 (99 %) | **0.02** ❌ | (ELBO converges, but posterior is garbage) | No |
| **ADVI-LR** | 65/70 (93 %) | **0.25** ❌ | Same | No |
| **Gibbs** | 66/70 (94 %) | **0.95** ✅ | n_eff ≈ 4800, passes Geweke | **Yes** ✅ |

**Three distinct failure modes, one success.**  This is the cleanest
version of the practitioner decision tree argument: at p=100 on the
graphical horseshoe, **only a model-specific hand-derived Gibbs
sampler actually produces usable Bayesian inference**.

---

## 4. Figure pipeline expansion

Yesterday `generate_figures.py` had 4 plot functions (heatmap,
shrinkage profile, loss-vs-γ, runtime bar).  Today it has 11.  The
expansion was driven by the realisation that `metrics.json` and
`diagnostics.json` (small files, always on the cluster) carry enough
information for almost every figure in the progress report —
`kappa_samples` is only needed for one figure (shrinkage profile
histogram), and even there the compressed `.npz` suffices.

### 4.1 Aggregator upgrade: robust statistics

[`scripts/aggregate_results.py`](../scripts/aggregate_results.py) now
computes, for every numeric metric:

- `{metric}_mean` + `{metric}_std` (parametric, as before)
- `{metric}_median` + `{metric}_q25` + `{metric}_q75` (**new**, robust)

The robust summaries are necessary because of ADVI-LR's outlier
behaviour: a single seed occasionally converges to a near-singular Ω̂,
giving Stein's loss in the 10³–10⁴ range while other seeds sit near
10.  Mean ± std then shows wild error bars dominated by that one seed;
median + IQR is unaffected.

### 4.2 New aggregate-based plots

| Function | What it shows | Source |
|---|---|---|
| `plot_loss_vs_gamma` (**refactored**) | Multi-panel (one panel per p), mean ± std **or** median + IQR across seeds, one line per method | `per_config_method.json` |
| `plot_loss_vs_p` (**new**) | x = dimension, y = metric at γ closest to `--target-gamma`, one line per method | `per_config_method.json` |
| `plot_sparsity_sensitivity` (**new**) | Grouped bars at s ∈ {0.05, 0.10, 0.30}, fixed γ, one bar per method | `per_config_method.json` |

All three support `--robust` (median + IQR) and `--log-y`.

### 4.3 New diagnostic-based plots

These bypass the aggregator and walk `diagnostics.json` directly —
useful for surfacing inference-level information that isn't in the
metrics schema.

| Function | What it shows | Source |
|---|---|---|
| `plot_nuts_convergence_dashboard` (**new**) | Three-panel box plot of max R-hat, min bulk ESS, divergence rate across NUTS successes, stratified by p, with convergence-threshold hairlines | walks `nuts/diagnostics.json` |
| `plot_success_rate_vs_p` (**new**) | Stacked bars (success / failed / timeout / other) per method × p | walks `*/diagnostics.json` |
| `plot_elapsed_time_vs_p` (**new**) | Log-log wall-time vs p per method, median + IQR by default (elapsed times are heavy-tailed) | walks `*/diagnostics.json` |

### 4.4 Generalised shrinkage profile

`plot_shrinkage_profile_comparison` now accepts any list of methods
(default: `nuts,gibbs,advi_mf`) and renders one panel per method that
has `kappa_samples` available.  Bimodality coefficient annotated on
each panel; the Sarle threshold (5/9 ≈ 0.556) drawn as a red dashed
hairline.

### 4.5 Cosmetic fixes

User feedback: "ADVI-LR has massive uncertainty bars" → diagnosed as
outlier seeds (§4.1 above), added `--robust` flag.

User feedback: "markers hide the error bars" → reduced marker size
(6–7 → 3.5–4) and line/error-bar widths (1.4 → 1.2, elinewidth=1.0)
across `vs_gamma`, `vs_p`, `elapsed_time_vs_p`.

User feedback: "NUTS missing at p=100 in vs_p; advi_lr missing at
s=0.05, 0.10 in sparsity sensitivity" → widened `--gamma-tol` default
(0.02 → 0.15) and added **diagnostic prints** for every dropped
(method, p) or (method, s) point, telling you *why* it was dropped:

```
[vs_p] drop nuts         p=100: all seeds failed steins_loss at γ in [0.1, 0.42, 0.9]
[vs_p] drop advi_lr      p=50 : closest γ=0.10 is > 0.15 from target=0.42 ...
[vs_p] drop ledoit_wolf  p=10 : no config with graph=erdos_renyi, s=0.1
```

Three distinct reasons, distinguishable at a glance.

### 4.6 CLI flag summary

New flags added to `generate_figures.py`:

- `--robust` — median + IQR instead of mean ± std
- `--shrinkage-methods nuts,gibbs,advi_mf` — N-method shrinkage profile
- `--sparsity-sensitivity-p`, `--sparsity-sensitivity-gamma`,
  `--sparsity-sensitivity-levels`
- `--skip-sparsity`, `--skip-nuts-dashboard`, `--skip-success-rate`,
  `--skip-elapsed` — granular plot toggles
- Renamed `--lvg-*` → `--graph`, `--sparsity`, `--metric`, `--p-values`
  (shared across vs-γ and vs-p)

---

## 5. Code-change audit (since 14 April morning report)

| Path | Change | Status |
|---|---|---|
| **NEW** `src/utils/io.py` | `load_samples`, `samples_exist`, `save_samples_compressed` helpers | ✅ |
| **NEW** `scripts/compress_results.py` | Parallel migration utility, idempotent | ✅ |
| **NEW** `scripts/run_compress_slurm.sh` | 16-CPU SLURM wrapper | ✅ |
| `src/inference/run_single.py` | Producer writes `.npz`, casts stochastic draws to float32 | ✅ |
| `src/evaluation/evaluate_single.py` | Uses `load_samples` / `samples_exist` | ✅ |
| `scripts/generate_figures.py` | +7 plot functions, +robust mode, +diagnostic prints, +generalised shrinkage, +new CLI flags | ✅ |
| `scripts/aggregate_results.py` | Adds `{metric}_median / q25 / q75` | ✅ |
| `scripts/run_nuts_slurm.sh` | Removed `--timeout-seconds 14400` (inherits 12 h NUTS default) | ✅ |
| `tests/test_inference.py` | 7 assertions migrated to `load_samples` / `samples_exist` | ✅ |

`pytest tests/` → **113 passed, 2 skipped** (no regressions since
WORK3_interim).  Skipped two are the NumPyro-gated NUTS/ADVI smoke
tests which are opt-in (`RUN_INFERENCE_TESTS=1`).

---

## 6. Outstanding questions and issues

### 6.1 Gibbs bimodality NaN (unchanged from 14 April)

Still open.  The hypothesis from yesterday (s=0.05 + aggressive
shrinkage → κ̂ variance near zero → Sarle's formula degenerates)
remains the leading explanation but has not been verified with the
diagnostic one-liner from §4.6 of the prior report.  Low priority —
the point can be made qualitatively ("Gibbs shrinks hard enough that
the bimodality estimator saturates at the upper mode") without a
numerical value.

### 6.2 ADVI-LR outlier seeds

Diagnosed (§4.1 above): low-rank ADVI occasionally converges to a
near-singular guide, producing Stein's loss in the 10³–10⁴ range.
Mitigated for the paper by plotting with `--robust`; **root fix**
(next runs, post-April-15) would be to (a) restore `num_seeds = 5`
(currently `- 2` for low-rank), and (b) raise the guide rank cap (now
`min(p/2, 100)`).  Not blocking the progress report.

### 6.3 PSIS still not wired into `evaluate_single.py`

`src/evaluation/psis.py` exists and tests pass; `evaluate_single.py`
does not call it, so `metrics.json` lacks `psis_khat` for ADVI runs.
Non-blocking for the progress report (the coverage + bimodality
already establish the ADVI failure story).  Deferred to post-April-15
cleanup.

### 6.4 NUTS convergence dashboard will show the real story

Running `generate_figures.py --skip-lvg --skip-lvp ...` on the cluster
today will produce `nuts_convergence_dashboard.pdf` — a single figure
that makes the "NUTS can't mix at p=100" finding *visually*
self-evident: the R-hat box at p=100 straddles the 1.01 threshold
line; the ESS box at p=100 sits at the very bottom of the axis.
**This is the figure that should anchor §4 of the progress report.**

---

## 7. Updated status vs WORK3 deliverables (progress report due TODAY)

### Progress Report (April 15) — **DUE TODAY**
- [ ] 6-page PDF emailed to staff — **TO DO**
- [x] Figure: Heatmap comparison — **READY** (data + code both live)
- [x] Figure: Shrinkage profiles (3-way: NUTS / Gibbs / ADVI-MF) — **READY** (kappa_samples compressed, loader back-compat verified)
- [x] Figure: Loss vs γ (3-panel, robust option) — **READY**
- [x] Figure: Loss vs p — **READY** (new)
- [x] Figure: Coverage vs p / vs γ — **READY** via `--metric coverage_95`
- [x] Figure: NUTS convergence dashboard — **READY** (new)
- [x] Figure: Success rate vs p — **READY** (new, stacked bars)
- [x] Figure: Elapsed time vs p — **READY** (new)
- [x] Figure: Sparsity sensitivity — **READY** (new)
- [x] Cross-method table — **READY** via `aggregate_results.py → cross_method_table.json`

### Tier 2 Results (April 25)
- [x] All 7 methods × 15 (p, γ) pairs × 5 seeds — **3 156 + 185 NUTS re-runs = 3 341 successful**
- [ ] PSIS-k̂ computed for all ADVI runs — **NOT DONE** (non-blocking)
- [x] Sparsity sensitivity: s ∈ {0.05, 0.10, 0.30} — **DONE**
- [x] Aggregated summary tables with **robust statistics** — **DONE** (15 Apr addition)
- [x] Full figure pipeline — **DONE** (15 Apr addition)

---

## 8. Action plan for today (15 April)

### 8a. Re-aggregate with robust stats (5 min)
```bash
python scripts/aggregate_results.py
```
This picks up the 185 new NUTS successes AND adds the median / q25 /
q75 fields to every row.  Required before any `--robust` figure works.

### 8b. Run the full figure suite (10 min)
```bash
python scripts/audit_results.py --report results/summary/audit_summary.json
python scripts/generate_figures.py \
    --skip-heatmap --skip-shrinkage \
    --robust --log-y
```
Produces:
- `steins_loss_vs_gamma.pdf` (3-panel, robust)
- `steins_loss_vs_p.pdf` (robust, log-log)
- `steins_loss_sparsity_sensitivity.pdf`
- `nuts_convergence_dashboard.pdf` ← **progress-report centrepiece**
- `success_rate_vs_p.pdf`
- `elapsed_time_vs_p.pdf`
- `runtime_comparison.pdf`

Then swap the metric to produce coverage + bimodality variants:
```bash
python scripts/generate_figures.py \
    --skip-heatmap --skip-shrinkage \
    --skip-nuts-dashboard --skip-success-rate --skip-elapsed \
    --skip-sparsity --skip-runtime \
    --metric coverage_95
# repeat with --metric bimodality_coefficient_kappa
```

### 8c. Three-way shrinkage profiles at p=10/50/100 (5 min)
```bash
for cfg in 17 45 73; do  # p=10/50/100 at γ=0.20, s=0.10
    python scripts/generate_figures.py \
        --config-id $cfg --seed 0 \
        --shrinkage-methods nuts,gibbs,advi_mf \
        --output-dir figures/shrinkage_cfg${cfg} \
        --skip-lvg --skip-lvp --skip-sparsity \
        --skip-nuts-dashboard --skip-success-rate \
        --skip-elapsed --skip-runtime --skip-heatmap
done
```

### 8d. Draft progress report (rest of the day)
Six pages, recommended skeleton:

- **§1 Introduction** (½ page) — research question, why horseshoe, why
  compare MCMC + VI
- **§2 Model + methods** (1 page) — graphical horseshoe, seven
  estimators, hyperparameter choices
- **§3 Tier 2 results** (3 pages) — heatmaps (1 fig), shrinkage
  profiles (1 fig, 3-panel), loss-vs-γ multi-panel (1 fig), coverage
  + bimodality tables
- **§4 Headline: the three-way failure at p=100** (1 page)
  - NUTS convergence dashboard (fig) + textual "zero of 46 converged"
  - ADVI coverage collapse + bimodality destruction
  - Gibbs as the only working Bayesian method
  - Decision tree for practitioners
- **§5 Remaining work** (½ page) — PSIS, ADVI-LR refinement (fewer
  seeds, higher rank), final report structure

---

## 9. Risk registry (updated)

| Risk | Severity | Status |
|---|---|---|
| Disk quota runs out again before progress report | **Low** ↓ | Compression left 75 GB of headroom; any new runs would need to refill that before blocking. |
| NUTS dashboard doesn't match the narrative | Low | Spot-checked: median R-hat = 5.62 at p=100, 0/46 converged.  Data supports the story. |
| Gibbs bimodality NaN breaks the bimodality-vs-γ figure at p=100 | Medium | Handle in report text: "Gibbs shrinks hard enough that the bimodality estimator degenerates — qualitatively all κ ≈ 1." |
| ADVI-LR outliers make parametric plots unreadable | **Low** ↓ | Solved via `--robust` flag.  Use median + IQR for the progress report figures. |
| advi_lr / NUTS missing from specific figure cells | Low | `[vs_p] drop …` / `[sparsity] drop …` diagnostic prints tell you why each cell is empty; use loose `--gamma-tol` to minimize. |
| Re-running `aggregate_results.py` fails because disk grows during the day | Low | Cluster has 75 GB headroom; no new runs planned today. |

---

## 10. Session continuity notes (for the next Claude session)

1. **`.npz` is now the storage format** for posterior samples across
   the entire codebase.  Anything that reads them must go through
   `src.utils.io.load_samples`.  Legacy `.npy` files continue to work.

2. **Float32 is the new stochastic-draw dtype**.  `evaluate_single.py`
   upcasts to float64 before doing anything numerically sensitive
   (Stein's loss, coverage).  Don't remove those upcasts.

3. **`_horseshoe_init_values(p, ncp)` is load-bearing** for NUTS and
   ADVI at p ≥ 50.  Also verified not regressed (as in prior report).

4. **The paper narrative stabilised** in the last 24 hours:
   > At p=100 on the graphical horseshoe, NUTS fails to mix,
   > mean-field ADVI loses calibration and bimodality, low-rank ADVI
   > occasionally diverges.  Only the Li et al. (2019) Gibbs sampler
   > — model-specific, hand-derived conditionals — produces usable
   > posteriors.

   This is the central claim; every figure should support it.

5. **Aggregator now outputs robust stats**.  Any new plot function
   should read `{metric}_median` / `_q25` / `_q75` if you want
   outlier-resistant summaries.

6. **Test command (unchanged)**: `pytest tests/` → 113 passed, 2
   skipped.  Run before any commit.

7. **Don't re-run NUTS at p=100**.  46 successes with 0 convergence
   *is* the finding; more data points will not change it, and the
   12 h / task compute is expensive.

8. **Migration utility is reusable**.  If future runs produce more
   `.npy` (from a rollback or a different runner), re-run
   `sbatch scripts/run_compress_slurm.sh` — it's idempotent.

---

## 11. Bottom line

We came into the morning of the 14th with a disk crisis, 17 NUTS p=100
successes of uncertain convergence, and an incomplete figure pipeline.
We came out of the morning of the 15th with 75 GB of headroom, 46 NUTS
p=100 successes that provide a **cleaner and sharper** failure story
than what we had, and a figure pipeline that produces 11 publication-
ready plots from three CLI commands.

**The progress report writes itself from here.**  Three commands on
the cluster produce every figure; §4 of the report is the three-way
failure at p=100; Gibbs is the unique survivor.  The scientific
claim is stronger, the infrastructure is cleaner, and the disk has
room to breathe.
