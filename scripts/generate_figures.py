"""Generate publication-quality figures from the aggregated summary tables.

Produces three primary figures for the progress report:

1. ``heatmap_comparison.pdf`` — 6-panel heatmap for a single config,
   showing Omega_true and each method's Omega_hat side by side.
2. ``shrinkage_profiles.pdf`` — the paper's key figure.  Side-by-side
   histograms of the posterior-mean shrinkage coefficients kappa_hat
   from NUTS and ADVI-MF for one config.  NUTS should be bimodal,
   ADVI-MF should be unimodal.
3. ``loss_vs_gamma.pdf`` — curves of Stein's loss (and F1) vs gamma =
   p/T, one line per method.

All figures use the plotting utilities in ``src/utils/plotting.py``
where available.

Usage
-----
    python scripts/generate_figures.py \
        --config-id 40 --seed 0 \
        --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = REPO_ROOT / "data" / "synthetic" / "configs" / "config_manifest.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "synthetic"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "synthetic"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "results" / "summary"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"

# The order here controls the panel layout for the heatmap figure.
HEATMAP_METHODS = ["sample_cov", "ledoit_wolf", "glasso", "nuts", "gibbs", "advi_mf", "advi_lr"]


def _load_config(manifest_path: Path, config_id: int) -> dict:
    with open(manifest_path) as f:
        configs = json.load(f)
    for c in configs:
        if int(c["config_id"]) == int(config_id):
            return c
    raise KeyError(f"config_id {config_id} not in manifest {manifest_path}")


def _seed_dir(config: dict, seed: int, data_root: Path) -> Path:
    from src.utils.configs import dir_name_seed
    return data_root / config["dir_path"] / dir_name_seed(seed)


def _method_dir(config: dict, seed: int, method: str, results_root: Path) -> Path:
    from src.utils.configs import dir_name_seed
    return results_root / config["dir_path"] / dir_name_seed(seed) / method


# ======================================================================
# Figure 1: heatmap comparison
# ======================================================================

def plot_heatmap_comparison(config, seed, data_root, results_root, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.plotting import plot_precision_heatmap

    seed_dir = _seed_dir(config, seed, data_root)
    Omega_true = np.load(seed_dir / "omega_true.npy")

    methods_present = []
    for m in HEATMAP_METHODS:
        md = _method_dir(config, seed, m, results_root)
        if (md / "omega_hat.npy").exists():
            methods_present.append(m)

    n_panels = 1 + len(methods_present)
    cols = min(3, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes).ravel()

    # Shared color scale based on the true Omega
    vmax = float(np.max(np.abs(Omega_true - np.diag(np.diag(Omega_true)))))

    plot_precision_heatmap(
        Omega_true,
        title=rf"True $\Omega_0$  (p={config['p']}, $\gamma$={config['gamma']:.2f}, s={config['sparsity']})",
        ax=axes[0],
        vmax=vmax,
    )

    for i, method in enumerate(methods_present, start=1):
        Omega_hat = np.load(_method_dir(config, seed, method, results_root) / "omega_hat.npy")
        plot_precision_heatmap(
            Omega_hat,
            title=method,
            ax=axes[i],
            vmax=vmax,
        )

    # Hide unused axes
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Precision-matrix heatmaps — config {config['config_id']}, seed {seed}",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"heatmap_comparison    -> {output_path}")


# ======================================================================
# Figure 2: shrinkage profile comparison
# ======================================================================

def plot_shrinkage_profile_comparison(
    config, seed, results_root, output_path, *,
    methods=("nuts", "gibbs", "advi_mf"),
    bins=40,
):
    """Side-by-side histograms of posterior-mean κ̂ for each method.

    One panel per method that has kappa_samples available.  Silently skips
    methods whose samples are missing (e.g. frequentist baselines have no
    kappa), so passing a longer list is safe.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation.shrinkage import bimodality_coefficient
    from src.utils.io import load_samples, samples_exist

    present = []
    for m in methods:
        d = _method_dir(config, seed, m, results_root)
        if samples_exist(d, "kappa_samples"):
            present.append((m, d))

    if not present:
        print(f"[shrinkage_profile] no kappa_samples available for any of "
              f"{list(methods)} at config={config['config_id']} seed={seed}")
        return

    n_panels = len(present)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (method, d) in zip(axes, present):
        k = load_samples(d, "kappa_samples").mean(axis=0)  # (n_offdiag,)
        b = bimodality_coefficient(k)
        style = _METHOD_STYLES.get(method, {})
        ax.hist(
            k, bins=bins, range=(0, 1),
            color=style.get("color", "#7f7f7f"),
            edgecolor="black", linewidth=0.4, alpha=0.85,
        )
        ax.set_title(f"{method}  (bimodality = {b:.3f})")
        ax.set_xlabel(r"$\hat\kappa_{ij}$")
        ax.axvline(5 / 9, color="red", linestyle=":", linewidth=1,
                   label="bimodality threshold" if ax is axes[0] else None)
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_ylabel("count")
    axes[0].legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(
        rf"Posterior shrinkage profile $\hat\kappa_{{ij}}$  "
        f"(config {config['config_id']}, seed {seed})",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"shrinkage_profiles    -> {output_path}")


# ======================================================================
# Figure 3 helpers: aggregated loss figures (no single seed picked)
# ======================================================================

# Canonical plotting order + deterministic color/marker per method.
_METHOD_STYLES = {
    "nuts":        {"color": "#1f77b4", "marker": "o"},
    "gibbs":       {"color": "#17becf", "marker": "s"},
    "advi_mf":     {"color": "#d62728", "marker": "^"},
    "advi_lr":     {"color": "#ff7f0e", "marker": "v"},
    "advi_fr":     {"color": "#bcbd22", "marker": "D"},
    "glasso":      {"color": "#2ca02c", "marker": "P"},
    "ledoit_wolf": {"color": "#9467bd", "marker": "X"},
    "sample_cov":  {"color": "#8c564b", "marker": "*"},
}
_DEFAULT_METHODS = (
    "nuts", "gibbs", "advi_mf", "advi_lr", "glasso", "ledoit_wolf", "sample_cov",
)


def _load_per_cm(summary_dir: Path) -> list[dict] | None:
    """Load ``per_config_method.json``; print a hint and return None if missing."""
    path = summary_dir / "per_config_method.json"
    if not path.exists():
        print(f"[figures] {path} not found; run aggregate_results.py first")
        return None
    with open(path) as f:
        return json.load(f)


def _filter_rows(
    per_cm, *, p=None, graph=None, sparsity=None, method=None,
):
    """Filter aggregated rows by exact match on p/graph/sparsity/method."""
    out = []
    for r in per_cm:
        if p is not None and r["p"] != p:
            continue
        if graph is not None and r["graph"] != graph:
            continue
        if sparsity is not None and abs(r["sparsity"] - sparsity) > 1e-9:
            continue
        if method is not None and r["method"] != method:
            continue
        out.append(r)
    return out


def _point_and_err(row, metric, robust):
    """Return (center, err_low, err_high) for a metric row.

    In parametric mode returns (mean, std, std) — symmetric bars.
    In robust mode returns (median, median-q25, q75-median) — asymmetric,
    the IQR half-widths about the median.  Returns ``None`` if the center
    is missing.
    """
    if robust:
        m = row.get(f"{metric}_median")
        if m is None:
            return None
        q25 = row.get(f"{metric}_q25")
        q75 = row.get(f"{metric}_q75")
        lo = (m - q25) if q25 is not None else 0.0
        hi = (q75 - m) if q75 is not None else 0.0
        return (m, max(lo, 0.0), max(hi, 0.0))
    else:
        m = row.get(f"{metric}_mean")
        if m is None:
            return None
        s = row.get(f"{metric}_std") or 0.0
        return (m, s, s)


# ======================================================================
# Figure 3: loss vs gamma (multi-panel by p, uncertainty across seeds)
# ======================================================================

def plot_loss_vs_gamma(
    summary_dir,
    output_path,
    *,
    graph="erdos_renyi",
    sparsity=0.10,
    metric="steins_loss",
    methods=_DEFAULT_METHODS,
    p_values=(10, 50, 100),
    log_y=False,
    robust=False,
):
    """One panel per p; x = γ, y = metric center with seed-level error bars.

    Parametric mode: mean ± std.
    Robust mode (``--robust``): median + IQR (asymmetric bars about the median),
    which is resistant to the occasional ADVI-LR blow-up where a single seed
    produces Stein's loss orders of magnitude larger than the rest.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_cm = _load_per_cm(summary_dir)
    if per_cm is None:
        return

    ncols = len(p_values)
    fig, axes = plt.subplots(
        1, ncols, figsize=(5.2 * ncols, 4.8), sharey=log_y,
    )
    if ncols == 1:
        axes = [axes]

    any_drawn = False
    for ax, p in zip(axes, p_values):
        for method in methods:
            rows = _filter_rows(
                per_cm, p=p, graph=graph, sparsity=sparsity, method=method,
            )
            rows.sort(key=lambda r: r["gamma"])
            xs, ys, err_lo, err_hi = [], [], [], []
            for r in rows:
                pt = _point_and_err(r, metric, robust)
                if pt is None:
                    continue
                y, lo, hi = pt
                xs.append(r["gamma"])
                ys.append(y)
                err_lo.append(lo)
                err_hi.append(hi)
            if not ys:
                continue
            style = _METHOD_STYLES.get(method, {})
            ax.errorbar(
                xs, ys, yerr=[err_lo, err_hi],
                marker=style.get("marker", "o"),
                color=style.get("color"),
                label=method, capsize=3, linewidth=1.2, markersize=3.5,
                elinewidth=1.0,
            )
            any_drawn = True
        ax.set_xlabel(r"$\gamma = p / T$")
        ax.set_title(f"p = {p}")
        ax.grid(alpha=0.3)
        if log_y:
            ax.set_yscale("log")

    if not any_drawn:
        print(f"[loss_vs_gamma] no data for graph={graph}, s={sparsity}, "
              f"metric={metric}, p in {p_values}")
        plt.close(fig)
        return

    axes[0].set_ylabel(metric.replace("_", " "))
    # Single legend outside the right panel.
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    stat_label = "median + IQR" if robust else "mean ± std"
    fig.suptitle(
        f"{metric.replace('_', ' ')} vs $\\gamma$  "
        f"(graph={graph}, s={sparsity}; {stat_label} across seeds)",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{metric}_vs_gamma      -> {output_path}")


# ======================================================================
# Figure 3b: loss vs p (fixed gamma, uncertainty across seeds)
# ======================================================================

def plot_loss_vs_p(
    summary_dir,
    output_path,
    *,
    graph="erdos_renyi",
    sparsity=0.10,
    target_gamma=0.10,
    gamma_tol=0.15,
    metric="steins_loss",
    methods=_DEFAULT_METHODS,
    p_values=(10, 50, 100),
    log_y=True,
    robust=False,
):
    """x = p, y = metric center with seed-level error bars, one line per method.

    For each ``p`` in ``p_values`` we pick the configured ``γ`` closest to
    ``target_gamma`` (within ``gamma_tol``) and plot that config's center.

    If a ``(method, p)`` point is dropped — either because no config is
    within tolerance, or because every seed returned a non-finite metric —
    we print a diagnostic line explaining which of the two reasons applies.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_cm = _load_per_cm(summary_dir)
    if per_cm is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))

    any_drawn = False
    for method in methods:
        xs, ys, err_lo, err_hi = [], [], [], []
        for p in p_values:
            rows_all = _filter_rows(
                per_cm, p=p, graph=graph, sparsity=sparsity, method=method,
            )
            if not rows_all:
                print(f"  [vs_p] drop {method:12s} p={p:<3d}: no config with "
                      f"graph={graph}, s={sparsity}")
                continue
            rows = [r for r in rows_all if _point_and_err(r, metric, robust) is not None]
            if not rows:
                # Configs exist but every seed failed (metric is None).
                gammas_available = sorted(r["gamma"] for r in rows_all)
                print(f"  [vs_p] drop {method:12s} p={p:<3d}: "
                      f"all seeds failed {metric} at γ in {gammas_available}")
                continue
            best = min(rows, key=lambda r: abs(r["gamma"] - target_gamma))
            if abs(best["gamma"] - target_gamma) > gamma_tol:
                gammas_available = sorted(r["gamma"] for r in rows)
                print(f"  [vs_p] drop {method:12s} p={p:<3d}: closest γ={best['gamma']:.3f} "
                      f"is > {gamma_tol} from target={target_gamma} "
                      f"(available γ: {gammas_available})")
                continue
            pt = _point_and_err(best, metric, robust)
            y, lo, hi = pt
            xs.append(p)
            ys.append(y)
            err_lo.append(lo)
            err_hi.append(hi)
        if not ys:
            continue
        style = _METHOD_STYLES.get(method, {})
        ax.errorbar(
            xs, ys, yerr=[err_lo, err_hi],
            marker=style.get("marker", "o"),
            color=style.get("color"),
            label=method, capsize=3, linewidth=1.2, markersize=4,
            elinewidth=1.0,
        )
        any_drawn = True

    if not any_drawn:
        print(f"[loss_vs_p] no data within gamma_tol={gamma_tol} of "
              f"target_gamma={target_gamma} for graph={graph}, s={sparsity}")
        plt.close(fig)
        return

    ax.set_xlabel("dimension $p$")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xticks(list(p_values))
    ax.set_xticklabels([str(p) for p in p_values])
    if log_y:
        ax.set_yscale("log")
    stat_label = "median + IQR" if robust else "mean ± std"
    ax.set_title(
        f"{metric.replace('_', ' ')} vs $p$  "
        f"($\\gamma \\approx$ {target_gamma}, graph={graph}, s={sparsity}; "
        f"{stat_label} across seeds)"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{metric}_vs_p          -> {output_path}")


# ======================================================================
# Figure 3c: sparsity sensitivity (grouped bars at fixed γ)
# ======================================================================

def plot_sparsity_sensitivity(
    summary_dir,
    output_path,
    *,
    p=50,
    graph="erdos_renyi",
    target_gamma=0.42,
    gamma_tol=0.15,
    metric="steins_loss",
    methods=_DEFAULT_METHODS,
    sparsities=(0.05, 0.10, 0.30),
    log_y=True,
    robust=False,
):
    """Grouped bar chart: x-groups = sparsity, bars = method, y = metric center.

    Shows whether the horseshoe's advantage depends on the underlying sparsity.
    Missing (method × sparsity) bars are logged with a reason.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_cm = _load_per_cm(summary_dir)
    if per_cm is None:
        return

    n_groups = len(sparsities)
    n_methods = len(methods)
    width = 0.8 / max(n_methods, 1)
    x_centers = np.arange(n_groups)

    fig, ax = plt.subplots(1, 1, figsize=(max(9, n_groups * 3.5), 5))

    any_drawn = False
    for i, method in enumerate(methods):
        ys, err_lo, err_hi = [], [], []
        for s in sparsities:
            rows_all = _filter_rows(per_cm, p=p, graph=graph, sparsity=s, method=method)
            if not rows_all:
                print(f"  [sparsity] drop {method:12s} s={s}: no config with "
                      f"p={p}, graph={graph}")
                ys.append(np.nan); err_lo.append(0.0); err_hi.append(0.0)
                continue
            rows = [r for r in rows_all if _point_and_err(r, metric, robust) is not None]
            if not rows:
                gammas_available = sorted(r["gamma"] for r in rows_all)
                print(f"  [sparsity] drop {method:12s} s={s}: "
                      f"all seeds failed {metric} at γ in {gammas_available}")
                ys.append(np.nan); err_lo.append(0.0); err_hi.append(0.0)
                continue
            best = min(rows, key=lambda r: abs(r["gamma"] - target_gamma))
            if abs(best["gamma"] - target_gamma) > gamma_tol:
                gammas_available = sorted(r["gamma"] for r in rows)
                print(f"  [sparsity] drop {method:12s} s={s}: closest "
                      f"γ={best['gamma']:.3f} is > {gamma_tol} from target={target_gamma} "
                      f"(available γ: {gammas_available})")
                ys.append(np.nan); err_lo.append(0.0); err_hi.append(0.0)
                continue
            y, lo, hi = _point_and_err(best, metric, robust)
            ys.append(y)
            err_lo.append(lo)
            err_hi.append(hi)
        if all(np.isnan(v) for v in ys):
            continue
        offsets = x_centers + (i - (n_methods - 1) / 2) * width
        style = _METHOD_STYLES.get(method, {})
        ax.bar(
            offsets, ys, width=width * 0.92,
            yerr=[err_lo, err_hi],
            label=method,
            color=style.get("color"),
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
        any_drawn = True

    if not any_drawn:
        print(f"[sparsity_sensitivity] no data for p={p}, γ≈{target_gamma}, graph={graph}")
        plt.close(fig)
        return

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"s = {s}" for s in sparsities])
    ax.set_ylabel(metric.replace("_", " "))
    if log_y:
        ax.set_yscale("log")
    stat_label = "median + IQR" if robust else "mean ± std"
    ax.set_title(
        f"Sparsity sensitivity — {metric.replace('_', ' ')}  "
        f"(p={p}, γ≈{target_gamma}, graph={graph}; {stat_label})"
    )
    ax.legend(loc="best", frameon=False, ncol=2)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"sparsity_sensitivity  -> {output_path}")


# ======================================================================
# Figure 4a: NUTS convergence dashboard (from diagnostics.json directly)
# ======================================================================

def _walk_diagnostics(results_root: Path, method: str):
    """Yield (diagnostics_dict, p, gamma, sparsity, graph, seed) tuples."""
    for p in Path(results_root).rglob(f"{method}/diagnostics.json"):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        # p, gamma, graph, sparsity are fields the runner writes into diagnostics.json
        yield d, d.get("p"), d.get("gamma"), d.get("sparsity"), d.get("graph"), d.get("seed")


def plot_nuts_convergence_dashboard(
    results_root,
    output_path,
    *,
    p_values=(10, 50, 100),
):
    """Three-column dashboard: R-hat, min_bulk_ess, divergence_rate distributions.

    One row per p; one violin/box per row.  Reads ``diagnostics.json``
    directly so it isn't dependent on aggregate_results.py.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = {p: {"rhat": [], "ess": [], "div": []} for p in p_values}
    for d, p, _, _, _, _ in _walk_diagnostics(results_root, "nuts"):
        if d.get("status") != "success":
            continue
        if p not in data:
            continue
        rh = d.get("max_rhat")
        es = d.get("min_bulk_ess")
        dr = d.get("divergence_rate")
        if rh is not None and np.isfinite(rh):
            data[p]["rhat"].append(float(rh))
        if es is not None and np.isfinite(es):
            data[p]["ess"].append(float(es))
        if dr is not None and np.isfinite(dr):
            data[p]["div"].append(float(dr))

    have_any = any(vals for p in p_values for vals in data[p].values())
    if not have_any:
        print(f"[nuts_dashboard] no NUTS diagnostics found under {results_root}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = [f"p={p}" for p in p_values]

    def _box(ax, key, title, ylabel, threshold=None, log_y=False):
        series = [data[p][key] for p in p_values]
        # Filter empty series so matplotlib doesn't error
        positions = []
        present_series = []
        present_labels = []
        for i, (s, lab) in enumerate(zip(series, labels)):
            if s:
                positions.append(i + 1)
                present_series.append(s)
                present_labels.append(lab)
        if present_series:
            bp = ax.boxplot(
                present_series, positions=positions, widths=0.6, showfliers=True,
                patch_artist=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#cfe2f3")
        ax.set_xticks(list(range(1, len(labels) + 1)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        if threshold is not None:
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
                       label=f"threshold = {threshold}")
            ax.legend(loc="best", fontsize=8, frameon=False)
        ax.grid(alpha=0.3, axis="y")

    _box(axes[0], "rhat", "max R-hat", r"$\hat R$ (max over sites)",
         threshold=1.01, log_y=True)
    _box(axes[1], "ess", "min bulk ESS", "min ESS", threshold=400, log_y=True)
    _box(axes[2], "div", "divergence rate",
         "divergences / samples", threshold=0.05, log_y=False)

    fig.suptitle("NUTS convergence dashboard (successful runs only)", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"nuts_convergence      -> {output_path}")


# ======================================================================
# Figure 4b: success rate vs p (stacked bars per method)
# ======================================================================

def plot_success_rate_vs_p(
    results_root,
    output_path,
    *,
    methods=_DEFAULT_METHODS,
    p_values=(10, 50, 100),
):
    """Stacked bars: per method × p, fraction success / failed / timeout.

    Highlights 'NUTS fails at p=100, Gibbs fills the gap'.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # method -> p -> Counter of statuses
    totals = {m: {p: {"success": 0, "failed": 0, "timeout": 0, "other": 0}
                  for p in p_values} for m in methods}
    for method in methods:
        for d, p, *_ in _walk_diagnostics(results_root, method):
            if p not in totals[method]:
                continue
            s = d.get("status", "other")
            if s not in ("success", "failed", "timeout"):
                s = "other"
            totals[method][p][s] += 1

    # Each bar = one (method, p).  Arrange with methods grouped.
    fig, ax = plt.subplots(1, 1, figsize=(max(10, 1.2 * len(methods) * len(p_values)), 5))

    bar_colors = {
        "success": "#4caf50",
        "failed":  "#e53935",
        "timeout": "#ffb300",
        "other":   "#9e9e9e",
    }

    x_positions = []
    xticklabels = []
    pos = 0
    group_width = len(p_values)
    gap = 0.6

    for method in methods:
        for p in p_values:
            counts = totals[method][p]
            total = sum(counts.values())
            if total == 0:
                bottom = 0
                pos += 1
                continue
            frac = {k: v / total for k, v in counts.items()}
            bottom = 0
            for key in ("success", "failed", "timeout", "other"):
                v = frac[key]
                if v == 0:
                    continue
                ax.bar(pos, v, bottom=bottom, width=0.85,
                       color=bar_colors[key],
                       label=key if (method == methods[0] and p == p_values[0] and bottom == 0 and key == "success")
                       else None)
                bottom += v
            x_positions.append(pos)
            xticklabels.append(f"{method}\np={p}")
            pos += 1
        pos += gap  # gap between method groups

    # Build the legend manually so each status appears once
    handles = [plt.Rectangle((0, 0), 1, 1, color=bar_colors[k]) for k in ("success", "failed", "timeout", "other")]
    ax.legend(handles, ["success", "failed", "timeout", "other"],
              loc="lower right", frameon=False)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(xticklabels, fontsize=8, rotation=60, ha="right")
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(0, 1.02)
    ax.set_title("Inference status by method × dimension")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"success_rate_vs_p     -> {output_path}")


# ======================================================================
# Figure 4c: elapsed time vs p
# ======================================================================

def plot_elapsed_time_vs_p(
    results_root,
    output_path,
    *,
    methods=_DEFAULT_METHODS,
    p_values=(10, 50, 100),
    robust=True,
):
    """x = p, y = elapsed_seconds (per-run), with error bars per method.

    Uses ``elapsed_seconds`` from each method's diagnostics.json.  Robust
    by default (median + IQR) since elapsed times are heavy-tailed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = {m: {p: [] for p in p_values} for m in methods}
    for method in methods:
        for d, p, *_ in _walk_diagnostics(results_root, method):
            if d.get("status") != "success":
                continue
            if p not in data[method]:
                continue
            t = d.get("elapsed_seconds")
            if t is None:
                t = d.get("elapsed_core_seconds")
            if t is None:
                continue
            try:
                data[method][p].append(float(t))
            except (TypeError, ValueError):
                pass

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    any_drawn = False
    for method in methods:
        xs, ys, err_lo, err_hi = [], [], [], []
        for p in p_values:
            vals = data[method][p]
            if not vals:
                continue
            vals_arr = np.asarray(vals)
            if robust:
                med = float(np.median(vals_arr))
                q25 = float(np.percentile(vals_arr, 25))
                q75 = float(np.percentile(vals_arr, 75))
                xs.append(p)
                ys.append(med)
                err_lo.append(max(med - q25, 0.0))
                err_hi.append(max(q75 - med, 0.0))
            else:
                mean = float(np.mean(vals_arr))
                std = float(np.std(vals_arr, ddof=1)) if len(vals_arr) > 1 else 0.0
                xs.append(p)
                ys.append(mean)
                err_lo.append(std)
                err_hi.append(std)
        if not ys:
            continue
        style = _METHOD_STYLES.get(method, {})
        ax.errorbar(
            xs, ys, yerr=[err_lo, err_hi],
            marker=style.get("marker", "o"),
            color=style.get("color"),
            label=method, capsize=3, linewidth=1.2, markersize=4,
            elinewidth=1.0,
        )
        any_drawn = True

    if not any_drawn:
        print(f"[elapsed_time_vs_p] no data found under {results_root}")
        plt.close(fig)
        return

    ax.set_xlabel("dimension $p$")
    ax.set_ylabel("elapsed seconds")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(list(p_values))
    ax.set_xticklabels([str(p) for p in p_values])
    stat = "median + IQR" if robust else "mean ± std"
    ax.set_title(f"Wall-clock time per run vs $p$  ({stat})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"elapsed_time_vs_p     -> {output_path}")


# ======================================================================
# Figure 4: runtime comparison
# ======================================================================

def plot_runtime_comparison(summary_dir, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_cm_path = summary_dir / "per_config_method.json"
    if not per_cm_path.exists():
        print(f"[runtime] {per_cm_path} not found")
        return
    with open(per_cm_path) as f:
        rows = json.load(f)

    # Not all metrics.json files contain elapsed timing in the aggregate.
    # We pull elapsed medians directly from the audit summary if present.
    audit_path = summary_dir.parent / "summary" / "audit_summary.json"
    if not audit_path.exists():
        print(f"[runtime] {audit_path} not found, skipping runtime figure")
        return
    with open(audit_path) as f:
        audit = json.load(f)

    methods = list(audit.get("per_method", {}).keys())
    means = [audit["per_method"][m].get("mean_elapsed") or 0 for m in methods]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.bar(methods, means, color="steelblue")
    ax.set_ylabel("Mean elapsed (seconds)")
    ax.set_yscale("log")
    ax.set_title("Mean runtime per method")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"runtime_comparison    -> {output_path}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures from WORK2 summaries."
    )
    parser.add_argument("--config-id", type=int, default=None,
                        help="Config to use for heatmap + shrinkage figures.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--graph", type=str, default="erdos_renyi",
        help="graph family to filter aggregated figures by.",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.10,
        help="ground-truth sparsity to filter aggregated figures by.",
    )
    parser.add_argument(
        "--metric", type=str, default="steins_loss",
        help="Metric to plot vs γ and vs p (must match "
             "<metric>_mean / <metric>_std in per_config_method.json).",
    )
    parser.add_argument(
        "--p-values", type=str, default="10,50,100",
        help="Comma-separated list of p values to plot.",
    )
    parser.add_argument(
        "--target-gamma", type=float, default=0.10,
        help="γ at which to plot the loss-vs-p figure (nearest match per p).",
    )
    parser.add_argument(
        "--gamma-tol", type=float, default=0.15,
        help="Maximum |γ − target_gamma| allowed when selecting the per-p point "
             "(and the sparsity-sensitivity per-group point).  Loose default so "
             "the nearest γ in the grid qualifies even if not an exact match.",
    )
    parser.add_argument(
        "--log-y", action="store_true",
        help="Log y-axis on loss-vs-γ (loss-vs-p uses log y by default).",
    )
    parser.add_argument(
        "--robust", action="store_true",
        help="Plot median + IQR instead of mean ± std (outlier-resistant).",
    )
    parser.add_argument(
        "--shrinkage-methods", type=str, default="nuts,gibbs,advi_mf",
        help="Comma-separated methods for the shrinkage-profile figure.",
    )
    parser.add_argument(
        "--sparsity-sensitivity-p", type=int, default=50,
        help="p value for the sparsity-sensitivity grouped-bar figure.",
    )
    parser.add_argument(
        "--sparsity-sensitivity-gamma", type=float, default=0.42,
        help="Target γ for the sparsity-sensitivity figure.",
    )
    parser.add_argument(
        "--sparsity-sensitivity-levels", type=str, default="0.05,0.10,0.30",
        help="Comma-separated sparsity levels on the x-axis.",
    )
    parser.add_argument(
        "--skip-heatmap", action="store_true",
        help="Skip the heatmap comparison figure.",
    )
    parser.add_argument(
        "--skip-shrinkage", action="store_true",
        help="Skip the shrinkage-profile comparison figure.",
    )
    parser.add_argument(
        "--skip-lvg", action="store_true",
        help="Skip the loss-vs-γ figure.",
    )
    parser.add_argument(
        "--skip-lvp", action="store_true",
        help="Skip the loss-vs-p figure.",
    )
    parser.add_argument(
        "--skip-sparsity", action="store_true",
        help="Skip the sparsity-sensitivity grouped-bar figure.",
    )
    parser.add_argument(
        "--skip-nuts-dashboard", action="store_true",
        help="Skip the NUTS convergence dashboard.",
    )
    parser.add_argument(
        "--skip-success-rate", action="store_true",
        help="Skip the success-rate-vs-p figure.",
    )
    parser.add_argument(
        "--skip-elapsed", action="store_true",
        help="Skip the elapsed-time-vs-p figure.",
    )
    parser.add_argument(
        "--skip-runtime", action="store_true",
        help="Skip the single-bar runtime comparison figure.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.config_id is not None and not args.skip_heatmap:
        config = _load_config(args.manifest, args.config_id)
        plot_heatmap_comparison(
            config, args.seed, args.data_root, args.results_root,
            args.output_dir / "heatmap_comparison.pdf",
        )

    if args.config_id is not None and not args.skip_shrinkage:
        config = _load_config(args.manifest, args.config_id)
        shrinkage_methods = tuple(
            m.strip() for m in args.shrinkage_methods.split(",") if m.strip()
        )
        plot_shrinkage_profile_comparison(
            config, args.seed, args.results_root,
            args.output_dir / "shrinkage_profiles.pdf",
            methods=shrinkage_methods,
        )

    p_values = tuple(int(x) for x in args.p_values.split(",") if x.strip())
    sparsity_levels = tuple(
        float(x) for x in args.sparsity_sensitivity_levels.split(",") if x.strip()
    )

    if not args.skip_lvg:
        plot_loss_vs_gamma(
            args.summary_dir,
            args.output_dir / f"{args.metric}_vs_gamma.pdf",
            graph=args.graph,
            sparsity=args.sparsity,
            metric=args.metric,
            p_values=p_values,
            log_y=args.log_y,
            robust=args.robust,
        )

    if not args.skip_lvp:
        plot_loss_vs_p(
            args.summary_dir,
            args.output_dir / f"{args.metric}_vs_p.pdf",
            graph=args.graph,
            sparsity=args.sparsity,
            target_gamma=args.target_gamma,
            gamma_tol=args.gamma_tol,
            metric=args.metric,
            p_values=p_values,
            robust=args.robust,
        )

    if not args.skip_sparsity:
        plot_sparsity_sensitivity(
            args.summary_dir,
            args.output_dir / f"{args.metric}_sparsity_sensitivity.pdf",
            p=args.sparsity_sensitivity_p,
            graph=args.graph,
            target_gamma=args.sparsity_sensitivity_gamma,
            gamma_tol=args.gamma_tol,
            metric=args.metric,
            sparsities=sparsity_levels,
            robust=args.robust,
        )

    if not args.skip_nuts_dashboard:
        plot_nuts_convergence_dashboard(
            args.results_root,
            args.output_dir / "nuts_convergence_dashboard.pdf",
            p_values=p_values,
        )

    if not args.skip_success_rate:
        plot_success_rate_vs_p(
            args.results_root,
            args.output_dir / "success_rate_vs_p.pdf",
            p_values=p_values,
        )

    if not args.skip_elapsed:
        plot_elapsed_time_vs_p(
            args.results_root,
            args.output_dir / "elapsed_time_vs_p.pdf",
            p_values=p_values,
            robust=True,
        )

    if not args.skip_runtime:
        plot_runtime_comparison(
            args.summary_dir,
            args.output_dir / "runtime_comparison.pdf",
        )


if __name__ == "__main__":
    main()
