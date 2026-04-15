**Config 17 (p=10, γ=0.20, s=0.10) — The validation anchor**

45 off-diagonal pairs, ~4–5 true edges. All three methods are now virtually identical: a dominant bar at κ ≈ 1.0 (42–43 of 45 pairs fully shrunk) and 2–3 outlier bars in the κ = 0.6–0.9 range (the true signal entries partially preserved). Bimodality coefficients are nearly indistinguishable: NUTS b=0.924, **Gibbs b=0.923**, ADVI-MF b=0.954.

Compare to the pre-fix Gibbs: it was a single spike at exactly κ=1.0 with bimodality=NaN. Now the Gibbs panel shows the same fine structure as NUTS — a handful of entries with κ < 1 that correspond to the true nonzero entries in Ω₀. The shapes are so similar you could swap the NUTS and Gibbs labels and nobody would notice. This is exactly what you want from two exact MCMC methods targeting the same posterior.

At p=10 the problem is too easy to differentiate the methods. The panel's role in the paper is to establish the baseline: "when D is small, all three methods agree."

---

**Config 45 (p=50, γ=0.20, s=0.10) — Gibbs now matches NUTS; ADVI subtly degrades**

1,225 off-diagonal pairs, ~122 true edges. NUTS (b=0.897) shows the classic horseshoe profile: a tall dominant mode at κ ≈ 0.95–1.0 (noise entries, ~1,000 of 1,225 pairs) and a secondary spread from κ ≈ 0.55 to 0.90 (signal entries, with some partially shrunk). The secondary mode is diffuse — not a sharp peak, but a visible spread of bars with counts of 5–25 each, stretching from about 0.55 to 0.90.

**Gibbs (b=0.910)** now looks almost identical to NUTS. The dominant mode at κ ≈ 1.0 is the same height (~1,010 counts). The secondary spread has the same range and shape — bars visible from κ ≈ 0.55 to 0.90 with comparable counts. The bimodality coefficient is actually *slightly higher* than NUTS (0.910 vs 0.897), suggesting Gibbs may produce marginally sharper separation between signal and noise. This is a remarkable match: two completely different MCMC algorithms (gradient-based joint proposals vs. coordinate-wise closed-form conditionals) converging to the same posterior distribution.

Before the fix, Gibbs at p=50 was a single bar at κ=1.0 with b=NaN and Stein's loss 30× worse than NUTS. **The glasso initialisation completely resolved the cold-start collapse.** The τ² chain now has nonzero sum-ratio from sweep 1, so the global shrinkage parameter explores its full posterior support instead of collapsing to 10⁻¹⁶.

ADVI-MF (b=0.864) still shows the dominant mode at κ ≈ 1.0 and a secondary spread — the bimodality is preserved. But two subtle differences are visible compared to NUTS/Gibbs: the dominant mode bar is taller and narrower (~1,130 counts vs ~1,000), meaning ADVI is pushing slightly more entries toward full shrinkage; and the secondary bars are concentrated at κ ≈ 0.85–0.95, with less mass below 0.80. ADVI's signal entries are being slightly over-shrunk compared to the exact MCMC methods. This is the mean-field approximation manifesting at the shrinkage level — not catastrophic, but measurably different.

---

**Config 73 (p=100, γ=0.20, s=0.10) — The paper's central figure, now with the full three-way comparison**

4,950 off-diagonal pairs, ~495 true edges. This is where everything comes together.

**NUTS (b=0.902)**: dominant mode at κ ≈ 0.95–1.0 with ~4,300 counts. Secondary spread from κ ≈ 0.55 to 0.90 with counts of 10–50 per bin, plus a visible tail down to κ ≈ 0.30. The horseshoe structure is intact — clear bimodal separation between the ~4,450 noise entries (κ near 1) and the ~500 signal entries (κ spread below 0.90). Note that NUTS at p=100 has R̂ = 5.62 and didn't converge, so this profile is from non-mixed chains. Despite that, the *marginal* distribution of κ̂ still looks bimodal — the chains individually capture the horseshoe structure even if they disagree on specific entries.

**Gibbs (b=0.898)**: this is the headline. The shape now matches NUTS almost perfectly. Dominant mode at κ ≈ 0.95–1.0 with ~4,100 counts. Secondary spread from κ ≈ 0.55 to 0.95, with a visible cluster of bars around κ ≈ 0.85–0.95 (counts ~100–350) and a thinner tail extending down to κ ≈ 0.55 with smaller counts. The bimodality coefficient (0.898) is within 0.004 of NUTS (0.902). **Gibbs at p=100 now produces the same shrinkage structure as NUTS, but from a sampler that actually converges** (coverage 0.95, ESS ~4,800, 50 min vs 7 hours). This is the "Gibbs fills the gap" result fully materialised.

Before the fix, Gibbs was a degenerate spike at κ=1.0 with Stein's loss 30× worse than NUTS. After the fix, it matches NUTS on shrinkage structure while being the only method that passes convergence diagnostics at this dimension.

**ADVI-MF (b=0.462)**: catastrophic collapse, unchanged from the pre-fix figures (the Gibbs fix doesn't affect ADVI). The entire distribution has shifted leftward to κ ≈ 0.40–0.55, with the main mass in two adjacent bins at κ ≈ 0.45 and κ ≈ 0.50 (counts ~3,000 and ~1,700). A small residual bar at κ ≈ 0.55 (~150 counts). Nothing above κ = 0.60. The bimodality coefficient (0.462) is well below the 5/9 = 0.556 threshold — officially unimodal. Mean-field ADVI is applying moderate, indiscriminate shrinkage to every entry: noise entries that should be at κ ≈ 1 are instead at κ ≈ 0.5 (half-shrunk, leaving nonzero residual), and signal entries that should be at κ ≈ 0 are also at κ ≈ 0.5 (over-shrunk, killing the signal). The horseshoe's defining "shrink or don't" property has been completely destroyed.

**Success Rate vs. p — The reliability story**

This stacked bar chart shows what fraction of runs succeeded, failed, or timed out for each (method, p) combination.

The frequentist methods (glasso, LW, sample_cov) are solid green across all p — 100% success, as expected for closed-form or convex optimization methods. ADVI-MF is similarly near-100% at every p. ADVI-LR is nearly perfect at p=10 and p=50 but shows a thin failure sliver at p=100 (roughly 5/70 missing, consistent with the 65/70 from the April 15 report).

Gibbs is near-perfect: 100% at p=10 and p=50, and ~94% at p=100 (66/70). The small failure fraction at p=100 is likely a handful of runs where the Schur complement PD rejection hit the max-attempts cap. This is an excellent reliability profile for an exact MCMC method.

NUTS is the outlier. At p=10, it's 100%. At p=50, there's one missing run (69/70). At p=100, roughly a third of runs failed or timed out (46/70 success, ~24 fail/timeout). The red and orange slivers are visually striking against the solid green bars of every other method. This figure should go in the paper alongside the convergence dashboard — it shows that NUTS has problems at p=100 *before* you even check whether the successful runs converged (which they didn't).

For the paper: "Of the seven methods, only NUTS exhibits substantial failure rates, reaching 34% at p=100. Among the successes, zero meet convergence criteria (§6.4). The Gibbs sampler maintains 94% success with proper convergence."

---

**Stein's Loss vs. p (fixed γ ≈ 0.1) — The dimension scaling story, now with the corrected Gibbs**

This is a completely different figure from the pre-fix version. The Gibbs line (cyan) has dropped from ~6–50 to sit right next to NUTS.

At p=10, all methods cluster between 0.1 and 0.5. NUTS and Gibbs are at the bottom (~0.15), nearly indistinguishable. ADVI-MF and ADVI-LR are slightly above (~0.2). Glasso, LW, and sample_cov are at ~0.25–0.5. The ordering is tight.

At p=50, the methods fan out. **NUTS and Gibbs remain nearly tied** at ~0.6. ADVI-MF sits just below them at ~0.5 (actually marginally better than NUTS at this single γ — this may be a seed effect or ADVI happening to land near the posterior mode). ADVI-LR is at ~5 with a massive IQR bar extending up to ~20 — the outlier-seed instability noted in the April 15 report. LW is at ~2.5, sample_cov at ~3. Glasso jumps to ~7.

At p=100, the separation is dramatic. **Gibbs (cyan) is now the clear winner at ~1.5** — substantially better than NUTS (blue, ~3.5). This is a striking result: the model-specific Gibbs sampler produces *better point estimates* than NUTS at p=100, not just faster ones. The explanation is that NUTS's chains haven't converged (R̂ = 5.6), so its posterior mean is averaging over non-mixed chains from different posterior regions, distorting the estimate. Gibbs, which actually converges, computes its posterior mean from a properly mixed single chain. LW and sample_cov are at ~5. ADVI-MF and ADVI-LR have jumped to ~15 and ~10 respectively. Glasso is worst at ~50.

**The Gibbs < NUTS result at p=100 is the most important new finding.** Before the fix, Gibbs was 30× worse than NUTS. After the fix, Gibbs is 2× *better*. The entire ranking has inverted. This isn't Gibbs being unusually good — it's NUTS being unreliable because its chains haven't mixed. The posterior mean of non-converged chains is not a consistent estimator.

---

**Stein's Loss vs. γ (three panels) — The paper's central figure, completely transformed**

**p=10 (left panel):** All methods rise gently from ~0.2 at γ=0.10 to ~2–3 at γ=0.90. NUTS and Gibbs track each other at the bottom of the pack. ADVI-MF shows the odd dip at high γ that we noted before (prior dominates at small T). The spread is roughly one order of magnitude, and no method is dramatically better or worse. This panel confirms: "at low dimension, method choice barely matters."

**p=50 (center panel):** This is where the story sharpens. NUTS (blue) and **Gibbs (cyan) now travel together** from ~0.6 at γ=0.10 to ~5–6 at γ=0.90, overlapping nearly perfectly across the entire γ range. This is the validation that two exact MCMC methods targeting the same posterior produce the same point estimates — exactly as theory predicts. Before the fix, Gibbs was a flat line at ~25; now it's indistinguishable from NUTS.

ADVI-MF (red) sits 1–2× above the MCMC pair at low γ and roughly tracks them as γ increases, though with a wider IQR at γ=0.90 (the massive bar dropping below 1 is a seed where ADVI happened to get lucky). Glasso (green) starts near NUTS at γ=0.10 but rises faster, reaching ~10–15 at γ=0.67. LW, sample_cov, and ADVI-LR form a cluster at ~5–30 at high γ, with sample_cov and ADVI-LR worst.

**p=100 (right panel):** The most dramatic change from the pre-fix figure. **Gibbs (cyan) is now the best method at every γ value.** It starts at ~1.5 at γ=0.10 and rises to ~10 at γ=0.90. NUTS (blue) tracks slightly above at ~3–10, consistent with non-converged chains giving noisier point estimates. The two MCMC methods are within 2× of each other across the board, but Gibbs is consistently lower.

ADVI-MF and ADVI-LR are at ~10–20 at most γ values, roughly comparable to the frequentist methods. Glasso is flat at ~30 (its CV-selected ρ doesn't adapt well to changing γ at high p). Sample_cov and LW rise steeply at high γ, reaching ~70–80.

**The ordering at p=100 is now: Gibbs < NUTS < ADVI-MF ≈ LW < ADVI-LR < sample_cov < glasso.** The Bayesian methods (when they converge) dominate the frequentist baselines, and within the Bayesian methods, the model-specific sampler outperforms the generic one because it actually mixes.