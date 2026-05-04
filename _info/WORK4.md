# Agent Prompt: WORK4 — From Benchmark to Paper (Real Data + Mechanistic Figures)

## Context

You are completing the **final phase** of a research project comparing MCMC vs.
Variational Inference for sparse Bayesian precision matrix estimation under the
graphical horseshoe prior. MIT 6.7830, Spring 2026.

**Timeline:**
- **Apr 15** — Progress report submitted.
- **May 5** (20 days) — Final report due (12 pages, ICML style).
- **Final class meetings** (early May) — Presentation (10–15 min).

**What's done (WORK1 + WORK2 + WORK3):**
- 1,680 synthetic datasets (84 configs × 20 seeds), validated.
- Full inference + evaluation pipeline; 3,329 successful runs.
- Tier 2 fully populated: 7 methods × 42 (p, γ, s) configs.
- Gibbs Block-1 conditional bug found and fixed; corrected re-run completed.
- 113 + 1 regression tests passing.
- Posterior samples compressed in-place (170 GB → 95 GB) via `.npz` + float32 cast.
- 11 figure functions in `scripts/generate_figures.py` (loss-vs-γ multi-panel,
  loss-vs-p, shrinkage profile, sparsity sensitivity, NUTS dashboard, success
  rate, elapsed time, etc.) all driven by `--robust` (median + IQR) by default.

**Headline scientific findings as of progress report:**
1. **NUTS structurally fails at p=100**: 0/50 of "successful" runs converge
   (median R̂ = 5.01, median bulk-ESS = 2). Funnel geometry at D≈10,000 is
   incompatible with HMC.
2. **ADVI-MF coverage catastrophe at p=100**: 0.01 vs. nominal 0.95.
   Bimodality coefficient on κ collapses from 0.89 (p=50) to 0.45 (p=100).
3. **Gibbs is the unique Bayesian survivor at p=100**: matches NUTS on point
   estimates *and beats it* (Stein's loss 2.86 vs.\ 3.00), maintains 0.99
   coverage, preserves bimodality (b=0.90), 8× faster than NUTS, only Bayesian
   method that converges by all standard diagnostics.

**Feedback received on the progress report:**
> "The progress report reads like a benchmark catalog. The professor wants a
> *paper* — a focused argument about *why* inference method X fails on model Y."

**What this prompt adds beyond WORK3:**
- **Method restriction**: report only `gibbs / advi_mf / glasso / nuts` in the
  final paper (drop `advi_lr / ledoit_wolf / sample_cov`).
- **Two mechanistic "why" figures**: shrinkage anatomy + posterior geometry.
- **Real data application**: Fama–French 48 industry portfolios at p=48,
  T=250 (γ≈0.19) — synthetic findings transfer to a setting with no ground
  truth. Evaluate via out-of-sample log-likelihood and GMV portfolio variance.
- **Final report restructure**: 7-section, 12-page, "argument-first" writing.

This is the phase that converts a benchmark into a paper.

---

## 1. Method Restriction (D2)

The 7-method grid was useful for proposal-writing and breadth. The final paper
focuses on **four methods**, each with a single-sentence justification:

| Method | Role in the paper |
|---|---|
| **Gibbs** | Gold-standard exact posterior at all p; the paper's recommended sampler |
| **ADVI-MF** | The VI method under interrogation; cleaner story than ADVI-LR |
| **Graphical lasso** | Frequentist sparse benchmark; same sparsity goal, different paradigm |
| **NUTS** | Second exact baseline at p≤50; included at p=100 only to *demonstrate the failure* |

**Dropped from reporting** (kept on disk for appendix / reviewer questions):
- `advi_lr` — restart/rank issues never fully resolved; story muddied
- `ledoit_wolf` — dense, not really competing on the same problem
- `sample_cov` — trivially dominated, adds nothing

**Implementation**: no data deletion. Edit `_DEFAULT_METHODS` in
`scripts/generate_figures.py` and pass an explicit `methods=` argument in any
table-generation script. Run `aggregate_results.py` once more to regenerate
summaries with the restricted set if needed.

```python
# scripts/generate_figures.py
_FINAL_PAPER_METHODS = ("gibbs", "advi_mf", "glasso", "nuts")
```

Add a `--paper-methods` flag that defaults to this tuple; the CLI's existing
`--methods` flag remains for ad-hoc multi-method plots.

---

## 2. Two Mechanistic Figures (D3)

These two figures earn the paper its keep. Both run on **existing on-disk
samples**; no new cluster compute is required.

### 2.1 Shrinkage Anatomy (κ vs. |error|, signal/null coloring)

**File:** `scripts/generate_figures.py` → new function `plot_shrinkage_anatomy()`

**Output:** `figures/shrinkage_anatomy.pdf` — single PDF, two panels
(Gibbs left, ADVI-MF right).

**Configuration:**
- Single representative `(p, γ, s, seed)` config — recommended **(p=50, γ=0.42,
  s=0.10, seed=0)** since it's the regime where ADVI-MF "looks fine" on the
  marginal κ histogram but is already degrading on calibration.
- Both panels share x-axis, y-axis, and legend.

**For each method panel:**
1. Load `kappa_samples.npz` → average across samples → `kappa_hat`, shape `(n_offdiag,)`.
2. Load `omega_hat.npy` → flatten upper-triangular → `omega_hat_offdiag`.
3. Load `omega_true.npy` from the corresponding data dir → `omega_true_offdiag`.
4. Compute `err = abs(omega_hat_offdiag - omega_true_offdiag)`.
5. Compute `is_signal = abs(omega_true_offdiag) > 1e-5`.
6. Scatter:
   - x = `kappa_hat` (range [0, 1])
   - y = `err` (log scale recommended)
   - color = blue (null), red (signal)
   - alpha = 0.5–0.7 to handle overplotting
   - marker size proportional to `|omega_true|` for signals (or just consistent dots; user choice)

**Axes:**
- Both panels share `xlim=(0, 1)`, `ylim=(1e-4, 1)` (log).
- Add a vertical hairline at κ=0.5 to mark the "shrink" decision boundary.

**Headline observation expected:**
- *Gibbs panel*: two clean clusters. Most points (blue, true-zero entries)
  cluster at high κ ≈ 0.95–1 with low error (≤ 1e-2). Signal points (red)
  cluster at lower κ ≈ 0.6–0.8 with similar low error. Visually obvious
  separation by κ.
- *ADVI-MF panel*: everything smeared into κ ≈ 0.4–0.6, with **higher error
  on both signal and null entries simultaneously**. Visual proof that
  moderate-everywhere shrinkage hurts both.

**CLI:**
```bash
python scripts/generate_figures.py \
    --anatomy-config-id 53 --anatomy-seed 0 \
    --skip-everything-except-anatomy
```

(Replace 53 with whichever config maps to the chosen `(p, γ, s)`.)

### 2.2 Posterior Geometry (funnel vs. ellipse)

**File:** `scripts/generate_figures.py` → new function `plot_posterior_geometry()`

**Output:** `figures/posterior_geometry.pdf` — single PDF, two panels (Gibbs
left, ADVI-MF right).

**Configuration:**
- Same config as F1 (p=50, γ=0.42, s=0.10, seed=0).
- Pick a **single (i, j) pair that's a true signal edge**.
  - Heuristic: choose the (i, j) with `|omega_true[i,j]|` median across
    nonzero entries (avoid extreme outliers that make the figure too easy).
  - The function should print the chosen (i, j) and `omega_true[i,j]` so the
    paper can cite them.

**For each method panel:**
1. Load `omega_samples.npz`, `lambda_samples.npz`.
2. Index out `omega_samples[:, i, j]` and `lambda_samples[:, k]` where `k` is
   the flat upper-triangular index for `(i, j)` (use `pair_to_flat` from
   `gibbs_runner._build_index_maps`).
3. Scatter the joint distribution in `(ω_ij, log λ_ij)` space.
4. Overlay the ground-truth value `omega_true[i,j]` as a vertical line.

**Axes:**
- Both panels share x-range (centred on `omega_true[i,j]`).
- Both panels share y-range (`log λ_ij`).
- Use `density_kws={"alpha": 0.4}` and ~5,000 points per panel.

**Headline observation expected:**
- *Gibbs panel*: classic horseshoe **funnel** — wide spread in ω at high λ,
  narrowing as λ → 0. Strong negative correlation between ω and log λ.
- *ADVI-MF panel*: **axis-aligned ellipse**. Mean-field can only represent a
  product of independent Gaussians (one in ω, one in log λ). The funnel is
  geometrically not in the variational family.

**Caption text (draft):**
> The Gibbs sampler's posterior (left) shows the classic horseshoe funnel:
> when λ_ij is small, the conditional variance of ω_ij collapses, producing a
> narrow neck. ADVI-MF (right) approximates this joint distribution as a
> product of independent Gaussians, yielding an axis-aligned ellipse that
> mismatches the funnel's geometry. This mismatch is the mechanism by which
> mean-field VI fails on global-local priors at moderate-to-high dimension.

**CLI:**
```bash
python scripts/generate_figures.py \
    --geometry-config-id 53 --geometry-seed 0 \
    --geometry-edge auto    # or i,j pair like "12,34"
```

### 2.3 Implementation budget

Both figures = ~120 lines of new code in `generate_figures.py`. Should be
added in a single PR with three new tests:
- One asserting `plot_shrinkage_anatomy` produces a PDF when called with a
  valid config.
- One asserting `plot_posterior_geometry` finds at least one signal edge.
- One asserting the chosen edge has non-trivial λ posterior (avoid degenerate
  cases).

---

## 3. Real Data Pipeline (D1: Fama–French 48)

The Fama–French 48 industry portfolios (FF48) are precooked daily returns
from Ken French's data library. Choosing FF48 over CRSP gives:
- No name-selection methodology to defend
- Exactly p=48, T=250 → γ ≈ 0.19, in the regime where the synthetic results
  predict ADVI-MF *should* still work
- Public, stable, citable
- Preprocessed and clean (no missing data, no corporate actions, etc.)

### 3.1 Data ingest

**File:** `scripts/preprocess_real_data.py` (new, ~80 lines)

**Source:**
```
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/48_Industry_Portfolios_daily_CSV.zip
```

**Pipeline:**
1. `wget` or `urllib.request` download the ZIP.
2. Unzip → CSV with two sections: average value-weighted returns + average
   equal-weighted returns. **Use value-weighted** (more standard).
3. Skip the header rows (the Ken French CSVs have multi-line headers and
   sub-section dividers; parse robustly).
4. Strip rows with any `-99.99` (missing-data sentinel).
5. Convert to float, divide by 100 (the file is in percent, e.g.\ `0.42` = 0.42%).
6. **Demean each column** (so the multivariate-normal-zero-mean assumption holds).
7. Optionally **rescale to unit variance per column** (helps numerical
   conditioning at small T; document this choice).
8. Save as `data/real/ff48/Y.npy` shape `(T, 48)`.
9. Save metadata: date range, mean/std per column, number of dropped rows.

**Sanity check** built into the script:
- Print the first 5 dates, last 5 dates, and shape.
- Print min/max/mean of each column.
- Verify no NaNs, no infinities.
- Verify symmetry of the sample covariance.

**Acceptance criteria:**
- `python scripts/preprocess_real_data.py` produces `data/real/ff48/Y.npy` of
  shape `(T, 48)` with `T ≥ 250` and no NaNs.
- `data/real/ff48/metadata.json` records the source URL, download date, and
  preprocessing choices.

### 3.2 Train/test split

**File:** `scripts/build_real_data_splits.py` (new, ~50 lines)

For OOS evaluation we need at least one (train, test) pair. Two options:

**Option A — Single split (simpler, faster):**
- Train: first 250 days (~1 trading year).
- Test: next 250 days.
- One call per method, one set of OOS metrics.

**Option B — Rolling windows (more rigorous, gives confidence intervals):**
- 12 monthly windows: train on 250 days ending at month-end, test on the next
  ~21 trading days.
- 12 method runs per (method, window), aggregate via mean ± std.

**Recommendation:** start with Option A to wire up the pipeline. If time
permits and Option A's results are clean, upgrade to Option B for the final
report's robustness section.

**Output:** A list of `(start_date, train_end, test_end, train_idx, test_idx)`
tuples saved as `data/real/ff48/splits.json`.

### 3.3 Inference adaptation

The existing `src/inference/run_single.py` assumes a WORK1-style data
directory with `omega_true.npy` etc. For real data this doesn't exist.

**Two options:**

**Option A (recommended): synthesise a fake `omega_true.npy`** for the data
loader's sake (the inference code never uses it; only `evaluate_single.py`
does). Then the real-data evaluator skips ground-truth metrics entirely and
only computes OOS metrics.

**Option B: refactor the data-loading interface** so `omega_true` is
optional. Cleaner but more invasive.

Go with Option A: write a 5-line shim in `preprocess_real_data.py` that
saves a placeholder `omega_true.npy = np.eye(48)` and a sentinel
`real_data: true` in `metadata.json`. Then in `evaluate_single.py`, branch on
that sentinel to skip Stein's loss / Frobenius / coverage / bimodality.

### 3.4 OOS evaluation metrics

**File:** `src/evaluation/holdout.py` (new, ~150 lines)

Three core metrics:

**1. Out-of-sample negative log-likelihood**
```python
def oos_nll(omega_hat, Y_test):
    """Average negative log-likelihood under N(0, omega_hat^{-1})."""
    p = omega_hat.shape[0]
    T = Y_test.shape[0]
    sign, logdet = np.linalg.slogdet(omega_hat)
    if sign <= 0:
        return float("inf")  # not PD
    quad = np.sum(Y_test @ omega_hat * Y_test) / T  # mean over rows
    return 0.5 * (p * np.log(2*np.pi) - logdet + quad)
```

Lower = better. Compares "how well the precision estimate explains the
held-out data."

**2. GMV portfolio out-of-sample variance**

The **global minimum variance** portfolio under estimated Ω is:
```
w_GMV = (Ω̂·1) / (1·Ω̂·1)
```

Under the true Σ₀, this portfolio has variance `w' Σ₀ w`. With held-out
data, we estimate Σ₀ by `(1/T) Y_test' Y_test` and report:
```
σ²_OOS = w_GMV' (Y_test' Y_test / T) w_GMV
```

Lower = better. The textbook intuition: better Ω̂ gives more accurate w_GMV
and lower realised variance on unseen data.

```python
def gmv_oos_variance(omega_hat, Y_test):
    p = omega_hat.shape[0]
    one = np.ones(p)
    omega_one = omega_hat @ one
    w = omega_one / (one @ omega_one)
    Sigma_test = (Y_test.T @ Y_test) / Y_test.shape[0]
    return float(w @ Sigma_test @ w)
```

**3. Edge-set stability across windows (Bayesian-only)**

For each method that produces a posterior, compute the 95%-credible-interval
edge set on each window. Stability = Jaccard similarity across consecutive
windows. High stability = the method is finding genuine sparsity structure,
not arbitrary thresholding.

```python
def edge_jaccard(edge_set_1, edge_set_2):
    s1, s2 = set(edge_set_1), set(edge_set_2)
    if not (s1 | s2): return 1.0
    return len(s1 & s2) / len(s1 | s2)
```

**4. Optional: condition number of Ω̂**

For finance applications, condition number of the estimated precision
matrix is a quality proxy (well-conditioned = numerically stable portfolio
weights). Report alongside.

### 3.5 Run on cluster

One SLURM array, `len(splits) × 3 methods` jobs. Per-job wall:
- Gibbs at p=48 ≈ 5 min
- ADVI-MF at p=48 ≈ 1 min
- Glasso at p=48 ≈ 1 sec

Total compute: <10 min per window × 12 windows × 3 methods ≈ 6 hours
serially, but parallelizable.

**File:** `scripts/run_real_data_slurm.sh` (new, mirrors `run_gibbs_slurm.sh`).

### 3.6 Aggregation and figures

**File:** `scripts/generate_figures.py` → new function
`plot_real_data_calibration()`.

**Output:** `figures/real_data_oos.pdf` — two-panel figure:
- Panel 1: OOS NLL by method (boxplot across rolling windows, or bar at fixed split).
- Panel 2: GMV OOS variance by method (same).

**Headline expected (synthetic prediction holds):**
- Gibbs: best OOS NLL, lowest GMV variance.
- ADVI-MF: close to Gibbs (γ≈0.19 is the easy regime).
- Glasso: competitive on GMV (sparsity helps GMV directly), but worse OOS NLL.

If the synthetic prediction *doesn't* hold — that's an interesting story too,
and worth a paragraph in §6.

---

## 4. Final Report Structure (12 pages)

The progress report's section structure is a good v0. The final report
should refactor toward an argument-first layout:

| § | Pages | Contents |
|---|---|---|
| 1 Introduction | 1.0 | Single research question: "*At what dimension does mean-field ADVI stop faithfully representing the horseshoe's shrinkage structure, and why?*" Conclude with a one-paragraph summary of findings. |
| 2 Model | 1.5 | Graphical horseshoe; shrinkage coefficient κ; **funnel geometry** (use F2 here, *not* in §5). |
| 3 Inference methods | 1.5 | Gibbs (exact, model-specific), ADVI-MF (fast, approximate), glasso (frequentist), NUTS (exact but limited to p≤50, included to demonstrate geometric failure). One paragraph each on *why included*. |
| 4 Experimental setup | 1.0 | Synthetic grid (now restricted to ER + the three sparsities). FF48 setup. Evaluation metrics. |
| 5 Synthetic results | 3.0 | Stein's loss vs. γ (3 panels, 4 methods). Shrinkage anatomy figure F1. NUTS convergence dashboard. Coverage + bimodality table. |
| 6 Real data | 1.5 | FF48 setup, OOS NLL + GMV variance comparison, edge stability across windows. |
| 7 Discussion | 1.5 | "Model-specific samplers as the only viable Bayesian tool at moderate p." Connection to course themes (funnel geometry → Lecture 10; mean-field compactness → Lecture 4; model-specific Gibbs → Lecture 8). Limitations + future work. |

**Hard constraint:** 12-page limit. Use `\paragraph{}` aggressively, prune
the `\subsection` count.

**Critical edits relative to the progress report:**
- *Drop* the 5-bullet "headline findings" list at the end of §5; weave the
  same findings into prose throughout.
- *Drop* the standalone "Plot Description for the Final Report" section
  (was a placeholder for the progress report; the final report's plots are
  inline).
- *Drop* the "Implementation Challenges" section (push to an appendix or
  a 2-line acknowledgement of the Gibbs Block-1 fix).
- *Add* the geometry figure F2 in §2 — this is the paper's most explanatory
  figure and belongs in the model section, not the results.
- *Add* the shrinkage anatomy figure F1 in §5 — the "why" replacement for
  the Gibbs/NUTS κ histograms.
- *Add* the real-data section §6.
- *Sharpen* §1's research question to one sentence.

---

## 5. Code Map (new files / modified files)

### New files
```
scripts/
  preprocess_real_data.py      # FF48 ingest + preprocessing
  build_real_data_splits.py    # train/test windows
  run_real_data_slurm.sh       # SLURM array for FF48 inference
src/evaluation/
  holdout.py                   # OOS NLL, GMV variance, edge stability
data/real/
  ff48/
    Y.npy                      # (T, 48)
    omega_true.npy             # placeholder = I_48 (real-data sentinel)
    metadata.json              # source URL, dates, preprocessing choices
    splits.json                # list of (train_idx, test_idx) tuples
```

### Modified files
```
scripts/
  generate_figures.py          # +plot_shrinkage_anatomy, +plot_posterior_geometry,
                               # +plot_real_data_calibration, +_FINAL_PAPER_METHODS,
                               # +--paper-methods CLI flag
  run_inference_single.py      # accept --data-dir data/real/...
  aggregate_results.py         # restrict to _FINAL_PAPER_METHODS by default
src/evaluation/
  evaluate_single.py           # branch on metadata.real_data sentinel; skip
                               # ground-truth metrics for real data
src/inference/
  run_single.py                # ensure handles real-data dir layout (read Y.npy,
                               # metadata.json without needing omega_true)
tests/
  test_holdout.py              # new: tests for oos_nll, gmv_oos_variance, edge_jaccard
  test_real_data_pipeline.py   # new: end-to-end smoke test on a tiny synthetic
                               # example masquerading as "real data"
```

### Files unchanged but re-run
```
scripts/aggregate_results.py   # re-run after method restriction
scripts/audit_results.py       # re-run for updated audit_summary.json
scripts/generate_figures.py    # re-run all plots with --paper-methods
```

---

## 6. Acceptance Criteria

A WORK4 task is "done" when the corresponding box below is ticked.

### Phase 0: decisions
- [x] D1: FF48 chosen as real-data source.
- [x] D2: 4 methods locked: gibbs, advi_mf, glasso, nuts.
- [x] D3: 2 mechanistic figures committed: shrinkage anatomy + posterior geometry.

### Phase 1: method restriction
- [ ] `_FINAL_PAPER_METHODS` constant defined; `--paper-methods` CLI flag works.
- [ ] All paper figures rendered with the 4-method default; backups of the
      pre-restriction PDFs kept under `figures/archive/`.
- [ ] `aggregate_results.py` outputs reflect the restricted set if filtered.
- [ ] `pytest tests/` still passes (113 + regression test = 114 + 2 skipped).

### Phase 2: mechanistic figures
- [ ] `plot_shrinkage_anatomy()` produces `figures/shrinkage_anatomy.pdf`.
- [ ] `plot_posterior_geometry()` produces `figures/posterior_geometry.pdf`.
- [ ] Chosen edge for F2 reported in the figure caption (with i, j, ω_true).
- [ ] Both new functions covered by smoke tests in `tests/test_figures.py`.

### Phase 3: FF48 pipeline
- [ ] `data/real/ff48/Y.npy` exists, shape `(T, 48)`, `T ≥ 250`, no NaNs.
- [ ] `data/real/ff48/metadata.json` records source URL, download date,
      preprocessing choices.
- [ ] `data/real/ff48/splits.json` records the train/test windows.
- [ ] `src/evaluation/holdout.py` covered by unit tests for all three metrics.
- [ ] Cluster array submitted; per-method × per-window run produces
      `metrics_holdout.json` next to the standard `metrics.json`.
- [ ] Aggregate OOS results reproduce or refute the synthetic prediction.
- [ ] `figures/real_data_oos.pdf` renders.

### Phase 4: writing
- [ ] `final_report.tex` skeleton with 7 sections, hard 12-page LaTeX cap.
- [ ] Each `\todo{}` block in the skeleton resolved.
- [ ] All figures referenced from text; no orphan figures.
- [ ] All numeric claims in text traceable to a verification command.
- [ ] Final draft compiled to `final_report.pdf` in 12 pages or fewer.

### Phase 5: polish
- [ ] One full editing pass with the team.
- [ ] Presentation slides drafted (10–15 min, 12–18 slides).
- [ ] Submission ready by **May 5, 2026**.

---

## 7. Timeline (Gantt)

```
Apr 15  ─┬── PROGRESS REPORT SUBMITTED
        │
Apr 16  ─┼── Phase 1: method restriction (1 day, all team in standup)
Apr 17  ─┤   ├── F1 shrinkage anatomy (Arturo)
Apr 18  ─┤   ├── F2 posterior geometry (Nick)
Apr 19  ─┤   └── FF48 ingest + preprocessing (Federico)
        │
Apr 20  ─┼── FF48 train/test splits + inference adaptation (Federico)
Apr 21  ─┤
Apr 22  ─┤   ├── F1, F2 finalised; review meeting
Apr 23  ─┤   ├── OOS metrics module (Federico)
        │   └── Draft §1–4 of final report (Arturo)
Apr 24  ─┼── Cluster run on FF48 (Federico)
        │
Apr 25  ─┤   Real-data aggregation + figures (Federico)
Apr 26  ─┤
Apr 27  ─┤   Draft §5 with new figures (Nick)
Apr 28  ─┼── Draft §6–7 (Federico, Arturo)
        │
Apr 29  ─┤
Apr 30  ─┼── Full draft v1 (all sections, all figures, may be over 12 pages)
        │
May 1   ─┤   Trim to 12 pages (all)
May 2   ─┤   Editing pass + slides
May 3   ─┼── Slides finalized; final report compile
May 4   ─┤   Buffer / final polish
May 5   ─┼── SUBMIT
```

---

## 8. Risk Register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| FF48 preprocessing reveals data quality issues | Low | Medium | Spend Apr 17 on exploratory QC; Ken French's data is normally pristine. |
| Posterior geometry figure F2 doesn't show the expected funnel for the chosen edge | Medium | Medium | Try 2–3 candidate edges; pre-screen by `λ_ij` posterior median in [0.3, 1.0]. |
| FF48 results contradict synthetic findings (e.g.\ ADVI-MF *better* than Gibbs OOS) | Low | High | This is itself a finding; discuss honestly in §6 and §7. |
| 12-page limit blown | Medium | Low | Hard cap from day 1; cut as you write. |
| Cluster quota issue (FF48 jobs fail to start) | Low | Medium | Compute footprint is tiny (~6 h serial total); jobs should slot in any time. |
| Real-data section reveals a bug in `evaluate_single.py`'s real-data branch | Medium | Medium | Test the real-data sentinel path on a synthetic dummy first. |
| F1 anatomy figure shows ADVI-MF doing fine at p=50 | Low | Low | Pre-pick p=100 if p=50 isn't dramatic enough; the failure is more visual at higher p. |

---

## 9. Connection to Course Themes (for §7 Discussion)

The final paper's discussion should explicitly connect findings to course
material:

- **Funnel geometry & HMC failure** → Lecture 10 (NUTS, Hamiltonian dynamics):
  the geometric failure mode at p=100 is exactly what Betancourt (2017) warns
  about for hierarchical models. The horseshoe's local-global structure
  creates a posterior that no leapfrog step size can navigate.
- **Mean-field compactness** → Lecture 4 (Variational Inference): Turner &
  Sahani's "compactness" problem manifests as the κ collapse — the
  mode-seeking nature of KL(q∥p) under-covers heavy tails, which the
  horseshoe has by design.
- **Model-specific Gibbs** → Lecture 8 (MCMC, Gibbs sampling): a hand-derived
  conditional sampler with 100% acceptance outperforms a generic gradient
  sampler on a model where the gradient geometry is hostile. This is the
  course's core "match algorithm to model" message.
- **Bayesian model evaluation** → out-of-sample log-likelihood + posterior
  predictive checks (real data section). PSIS-k̂ if time permits.

---

## 10. What This Prompt Doesn't Cover

Out of scope for WORK4 (acceptable to drop or defer):
- Block-diagonal graph structure sensitivity (was in WORK3 §13).
- Wire PSIS-k̂ into `evaluate_single.py` (was in WORK3 §13). Can be
  computed post-hoc from the archived ADVI samples on the Mac if a
  reviewer asks.
- ADVI-LR re-run with more restarts. Confirmed dropped from final paper.
- Multiple real-data sources (e.g.\ S&P 500 sectors). FF48 alone is enough
  for one section.
- Tier 3 from the original plan (block-diagonal, larger p). Final paper
  uses Tier 2 only.

---

## 11. Glossary (for new team members reading this prompt cold)

- **Tier 2**: synthetic experimental grid restricted to Erdős–Rényi graphs and
  three sparsity levels {0.05, 0.10, 0.30}; 42 (p, γ, s) configs total.
- **NCP**: non-centred parameterisation, ω_ij = z_ij · λ_ij · τ.
- **κ_ij**: shrinkage coefficient `1 / (1 + λ_ij² τ²)`; the horseshoe's
  signature diagnostic.
- **Bimodality coefficient b**: Sarle's coefficient on the distribution of
  `{κ̂_ij}`; b > 5/9 ≈ 0.556 indicates "shrink or don't" structure preserved.
- **OOS**: out-of-sample.
- **GMV**: global minimum variance (a portfolio construction).
- **FF48**: Fama–French 48 industry portfolios (daily return matrix).

---

## 12. The One-Paragraph Pitch (for §1 of the final report)

To anchor the writing: the paper's argument in one paragraph.

> *Sparse Bayesian precision matrix estimation under the graphical horseshoe
> prior poses a sharp inference challenge: the prior's local-global structure
> creates a funnel-shaped posterior that no off-the-shelf inference method
> handles well at moderate dimension. We show that, for p ≤ 50, both
> gradient-based MCMC (NUTS) and mean-field variational inference (ADVI-MF)
> recover the horseshoe's bimodal shrinkage structure with near-nominal
> calibration, but the mechanism diverges at p=100: NUTS's chains fail to
> mix (median R̂ = 5.0), while ADVI-MF's coverage collapses to 0.01 and the
> shrinkage signature is destroyed. Only the model-specific Gibbs sampler of
> Li, Craig, and Bhadra (2019), with hand-derived conditionals and 100%
> acceptance, produces calibrated, bimodal posteriors at p=100, doing so 8×
> faster than NUTS's non-convergent compute. We trace the dual failure to a
> single geometric cause — the funnel — and present a posterior-geometry
> figure that shows precisely why mean-field cannot represent it. Real-data
> experiments on the Fama–French 48 industry portfolios at γ ≈ 0.19 confirm
> the synthetic findings: model-specific samplers are the only reliable tool
> for graphical horseshoe inference at moderate p.*
