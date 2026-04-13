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
    config, seed, results_root, output_path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nuts_kappa_path = _method_dir(config, seed, "nuts", results_root) / "kappa_samples.npy"
    advi_kappa_path = _method_dir(config, seed, "advi_mf", results_root) / "kappa_samples.npy"
    if not nuts_kappa_path.exists() or not advi_kappa_path.exists():
        print(f"[shrinkage_profile] missing kappa samples; skipping "
              f"(nuts={nuts_kappa_path.exists()}, advi={advi_kappa_path.exists()})")
        return

    nuts_k = np.load(nuts_kappa_path).mean(axis=0)  # (n_offdiag,)
    advi_k = np.load(advi_kappa_path).mean(axis=0)

    from src.evaluation.shrinkage import bimodality_coefficient
    from src.utils.plotting import plot_shrinkage_profile

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    plot_shrinkage_profile(nuts_k, advi_k, ax=axes)

    b_nuts = bimodality_coefficient(nuts_k)
    b_advi = bimodality_coefficient(advi_k)
    axes[0].set_title(f"NUTS  (bimodality = {b_nuts:.3f})")
    axes[1].set_title(f"ADVI-MF  (bimodality = {b_advi:.3f})")

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
# Figure 3: loss vs gamma
# ======================================================================

def plot_loss_vs_gamma(
    summary_dir, output_path,
    p=50, graph="erdos_renyi", sparsity=0.10,
    metric="steins_loss",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lvg_path = summary_dir / "loss_vs_gamma.json"
    if not lvg_path.exists():
        print(f"[loss_vs_gamma] {lvg_path} not found; run aggregate_results.py first")
        return
    with open(lvg_path) as f:
        lvg = json.load(f)

    curves = [
        c for c in lvg
        if c["p"] == p and c["graph"] == graph and c["sparsity"] == sparsity
    ]
    if not curves:
        print(f"[loss_vs_gamma] no curves match p={p}, graph={graph}, s={sparsity}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    for c in curves:
        method = c["method"]
        xs = c["gammas"]
        ys = c.get(mean_key, [None] * len(xs))
        es = c.get(std_key, [None] * len(xs))
        xs_plot = [x for x, y in zip(xs, ys) if y is not None]
        ys_plot = [y for y in ys if y is not None]
        es_plot = [e for e, y in zip(es, ys) if y is not None]
        if not ys_plot:
            continue
        ax.errorbar(xs_plot, ys_plot, yerr=es_plot, marker="o", label=method,
                    capsize=3, linewidth=1.4)

    ax.set_xlabel(r"$\gamma = p / T$")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(
        f"{metric} vs $\\gamma$  "
        f"(p={p}, graph={graph}, s={sparsity})"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"loss_vs_gamma         -> {output_path}")


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
        "--lvg-p", type=int, default=50,
        help="p for the loss-vs-gamma plot.",
    )
    parser.add_argument(
        "--lvg-graph", type=str, default="erdos_renyi",
        help="graph for the loss-vs-gamma plot.",
    )
    parser.add_argument(
        "--lvg-sparsity", type=float, default=0.10,
        help="sparsity for the loss-vs-gamma plot.",
    )
    parser.add_argument(
        "--lvg-metric", type=str, default="steins_loss",
        help="Metric to plot vs gamma.",
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
        help="Skip the loss-vs-gamma figure.",
    )
    parser.add_argument(
        "--skip-runtime", action="store_true",
        help="Skip the runtime comparison figure.",
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
        plot_shrinkage_profile_comparison(
            config, args.seed, args.results_root,
            args.output_dir / "shrinkage_profiles.pdf",
        )

    if not args.skip_lvg:
        plot_loss_vs_gamma(
            args.summary_dir,
            args.output_dir / f"{args.lvg_metric}_vs_gamma.pdf",
            p=args.lvg_p, graph=args.lvg_graph, sparsity=args.lvg_sparsity,
            metric=args.lvg_metric,
        )

    if not args.skip_runtime:
        plot_runtime_comparison(
            args.summary_dir,
            args.output_dir / "runtime_comparison.pdf",
        )


if __name__ == "__main__":
    main()
