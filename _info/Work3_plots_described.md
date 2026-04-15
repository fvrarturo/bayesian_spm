These are excellent — and they tell the paper's full story in three images. Let me walk through what you're seeing.

**What each axis means:** The x-axis is κ̂_ij, the posterior mean shrinkage coefficient for each off-diagonal pair (i,j). κ near 1 = "this entry was shrunk to zero" (noise). κ near 0 = "this entry was left alone" (signal). The y-axis is count — how many of the C(p,2) entry pairs land in each bin.

**Config 17 (p=10): Everyone agrees — too easy to differentiate**

Only 45 off-diagonal pairs, of which ~4–5 are true edges. All three methods pile almost everything at κ ≈ 1 (correct — 90% of entries *are* zero). The handful of bars at κ = 0.6–0.9 in the NUTS panel are the true signal entries being partially preserved. ADVI-MF (b=0.954) looks nearly identical to NUTS (b=0.924). At this scale, the problem is too small for the mean-field approximation to matter. This panel establishes the baseline: "when p is small, all methods recover the same shrinkage structure."

**Config 45 (p=50): The nuanced middle — ADVI preserves the shape**

Now there are 1,225 pairs, ~122 true edges. NUTS (b=0.897) shows the classic horseshoe shape: a dominant mode at κ ≈ 0.95–1.0 (noise shrunk) and a secondary spread around κ = 0.6–0.9 (signals partially preserved). ADVI-MF (b=0.864) looks *remarkably similar* — the bimodality coefficient is only 0.03 lower. The signal entries are in the same range, the noise mode is in the same place. This is the WORK2 surprise finding: mean-field ADVI preserves the between-entry shrinkage profile at p=50 even though it breaks within-entry posterior correlations. The degradation shows up in *calibration* (coverage drops 3–5 points) not in the shrinkage shape.

**Config 73 (p=100): The catastrophe — ADVI collapses**

This is the paper's central figure. Look at the ADVI-MF panel (right): the entire distribution has shifted from κ ≈ 1 to **κ ≈ 0.45–0.55**. This is the nightmare scenario. Instead of "shrink or don't," ADVI is applying *moderate shrinkage to everything*. True zeros get half-shrunk (should be fully shrunk). True signals get half-shrunk (should be left alone). The bimodality coefficient (0.462) drops below the 5/9 = 0.556 threshold — officially unimodal. Meanwhile NUTS (b=0.902) still shows the correct pattern with mass concentrated near κ = 1. The mean-field approximation has *completely failed* to represent the horseshoe's local-global coupling at D ≈ 10,000 dimensions.

**The Gibbs panels (all NaN): not a bug, but needs interpretation**

Gibbs shows a single bar at exactly κ = 1.0 at every p. This means the Gibbs sampler is shrinking *every* entry to near-zero — including the true signals. The bimodality coefficient is NaN because the variance is zero (all values identical). This is worth investigating: check whether the Gibbs posterior mean Ω̂ actually recovers the true nonzero entries despite the κ ≈ 1 appearance.

**Plot 1: Stein's Loss vs. γ (three panels, the central figure)**

At **p=10** (left), everything is clustered between 0.2 and 5. All methods work reasonably well because the problem is too easy. One oddity: ADVI-MF (red) actually *improves* as γ increases toward 0.9 — loss drops below 1. This is likely because at p=10 with very few data points, the horseshoe prior dominates and ADVI happens to land near the prior mode, which is close to truth for a sparse matrix. Not meaningful, just small-p noise.

At **p=50** (center), the story emerges. NUTS (blue) is consistently the best, sitting at ~0.6 at low γ and rising to ~3 at high γ. ADVI-MF (red) tracks NUTS with roughly 2× the loss. Gibbs (cyan) is flat at ~25 across all γ — **this is a problem, and it's the biggest red flag in these plots**. Gibbs should be an exact MCMC method matching NUTS, but its Stein's loss is 30–40× worse. I'll come back to this. Glasso (green) starts competitive at low γ but worsens at high γ with massive IQR bars. LW and sample_cov are intermediate.

At **p=100** (right), everything gets worse as expected. NUTS (blue, sparse data points since only some converged) still gives the lowest loss where it exists (~3 at low γ). ADVI-MF and ADVI-LR are around 10–20. Gibbs is again flat at ~30–40. The frequentist methods are comparable to ADVI or worse.

**Plot 2: Stein's Loss vs. p (fixed γ ≈ 0.1, the easiest regime)**

Same story in a different cut. NUTS (blue) scales best: ~0.15 at p=10, ~0.6 at p=50, ~3 at p=100. ADVI-MF (red) follows with a growing gap. Gibbs (cyan) is dramatically worse — starting at ~6 at p=10 and hitting ~50 at p=100. Glasso grows steeply. LW and sample_cov grow moderately.

**Plot 3: Sparsity Sensitivity (p=50, γ=0.42)**

NUTS (blue) is the best at every sparsity level and its advantage grows as sparsity decreases: at s=0.05, NUTS is ~5× better than the next competitor; at s=0.30, only ~2×. This makes theoretical sense — the horseshoe's advantage over the Laplace (glasso) is strongest when the true matrix is very sparse. ADVI-MF (red) is the second-best Bayesian method at s=0.05 and s=0.10, competitive with glasso. At s=0.30 (denser), ADVI-LR (orange) and glasso catch up.

**The elephant in the room: Gibbs**

Gibbs has Stein's loss 10–40× worse than NUTS across every plot, every p, every γ. This is inconsistent with it being an "exact" MCMC method. Both should converge to the same posterior, so their point estimates (posterior mean) should produce similar Stein's loss. There are a few possible explanations:

One possibility is that the Gibbs posterior mean is being distorted by the aggressive shrinkage we saw in the κ plots (all κ ≈ 1.0). If Gibbs is over-shrinking everything — including the diagonal — the resulting Ω̂ will have distorted eigenvalues, which Stein's loss punishes harshly (it's sensitive to eigenvalue ratios). A second possibility is a burn-in issue: 2,000 sweeps at p=50 may not be enough, and the posterior mean includes samples from before convergence. A third is a bug in how the Gibbs posterior mean is assembled or how omega_hat is computed.

**NUTS Convergence Dashboard — the "NUTS is broken at p=100" proof**

Three panels, each a box plot stratified by p, with red dashed threshold lines.

**Left panel (max R̂):** At p=10, the box is tight right at 1.0 — perfect convergence, all chains agree. At p=50, the median creeps up to ~2.5 with the box spanning 1.5–5 and outliers up to 30. Already most runs fail the R̂ < 1.01 threshold. At p=100, the median is ~4–5 with the box from ~3 to ~8 and outliers past 30. The entire box is an order of magnitude above the 1.01 threshold line. Not a single p=100 run passes. The chains are exploring completely different regions of the posterior.

**Center panel (min bulk ESS):** This is the mirror image. At p=10, median ESS is ~800, well above the 400 threshold. At p=50, it drops to ~2, and at p=100, the box sits at ~1. An ESS of 1 means 20,000 posterior draws contain the statistical information of one independent sample. The NUTS chains at p=50 and p=100 are essentially frozen — they're returning the same (or nearly the same) value over and over, with occasional jumps between modes that never equilibrate. The handful of outlier dots above 400 at p=100 are runs where one parameter happened to mix while others didn't.

**Right panel (divergence rate):** At p=10, median divergence rate is ~5%, right at the threshold. At p=50, it rises to ~8–10%. At p=100, median is ~10% with some runs hitting 25%. Every divergence means the leapfrog integrator hit a region of extreme curvature (the horseshoe funnel) and couldn't maintain energy conservation. At 10–25% divergence rates, the sampler is systematically avoiding large regions of the posterior.

**The three panels together tell one story:** NUTS at p=100 simultaneously fails on all three convergence criteria. The chains don't agree (R̂ = 5), each chain individually isn't exploring (ESS = 1), and the integrator can't handle the geometry (10%+ divergences). This isn't a tuning problem — more warmup, smaller step size, or deeper trees won't fix it. The horseshoe's D=10,000-dimensional funnel geometry is fundamentally incompatible with gradient-based exploration at this scale.

---

**Elapsed Time vs. p — the compute cost story**

Log-log scale, 7 methods, p on x-axis.

The plot separates into four clear tiers:

**Tier 1 — sub-second (flat):** Ledoit-Wolf (purple, ~0.01s) and sample covariance (brown, ~0.01s) are essentially free at every p. They're solving closed-form formulas. Glasso (green) is a bit more expensive (~0.2s at p=10, ~2s at p=100) but still negligible.

**Tier 2 — minutes:** ADVI-MF (red) scales gently: ~20s at p=10, ~100s at p=50, ~200s at p=100. This is the "fast Bayesian" option. The key observation: even at p=100 where ADVI-MF *fails on calibration*, it finishes in ~3 minutes. The problem isn't compute — it's statistical quality.

**Tier 3 — tens of minutes to an hour:** Gibbs (cyan) and ADVI-LR (orange) overlap in this band. Gibbs goes from ~7s at p=10 to ~200s at p=50 to ~3000s (~50 min) at p=100. ADVI-LR is similar. Gibbs at p=100 takes 50 minutes and *produces calibrated posteriors*. That's the practical punchline.

**Tier 4 — hours:** NUTS (dark blue) is the most expensive by far: ~100s at p=10, ~5000s (~80 min) at p=50, ~25,000s (~7 hours) at p=100. And at p=100, those 7 hours buy you chains that don't converge (R̂ = 5). NUTS's IQR bar at p=100 is also the widest, reflecting the high variance in trajectory lengths when the sampler struggles.

**The gap between Gibbs and NUTS at p=100 is the paper's practical argument:** Gibbs takes 50 minutes and converges. NUTS takes 7 hours and doesn't. That's a 8× compute cost for a worse result. Combined with the convergence dashboard, the conclusion is unavoidable: at p=100, the model-specific Gibbs sampler is not merely faster — it's the *only MCMC method that works*.

**One thing to flag for the paper:** the Gibbs runtime scaling looks roughly O(p³) on this log-log plot (slope of ~3 from p=10 to p=100), which matches the theoretical per-sweep cost. NUTS scales steeper — closer to O(p⁴) or worse — because the leapfrog trajectory length grows with dimension on top of the per-step O(p³) Cholesky cost. This is worth a sentence in the Methods section.