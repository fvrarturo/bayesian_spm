"""Aggregate per-seed WORK2 metrics into per-config-method summaries.

Walks ``results/synthetic/`` and for each ``(config_id, method)`` pair
collects every ``metrics.json`` across seeds, computing the mean and
std (across successful seeds) of every scalar metric.  Results are
written under ``results/summary/``.

Three outputs:

1. ``per_config_method.json`` — list of dicts, one per (config × method).
2. ``cross_method_table.json`` — pivot table: per config, every method
   as a row with the mean ± std of the headline metrics.
3. ``loss_vs_gamma.json`` — per (p, graph, sparsity, method), the
   mean/std of each metric as a function of gamma.  Feeds the loss-vs-γ
   figure generator.

Usage
-----
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --output-dir results/summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.configs import dir_name_seed  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "data" / "synthetic" / "configs" / "config_manifest.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "synthetic"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "results" / "summary"

ALL_METHODS = ["nuts", "advi_mf", "advi_fr", "glasso", "ledoit_wolf", "sample_cov"]

# Numeric metrics we aggregate (mean/std across seeds).  Others pass through
# as the mode / last value.
NUMERIC_FIELDS = [
    "steins_loss",
    "frobenius_loss",
    "frobenius_loss_relative",
    "spectral_loss",
    "trace_error",
    "eigenvalue_mse",
    "condition_number_hat",
    "condition_number_true",
    "gmv_weight_norm",
    "oracle_gmv_weight_norm",
    "gmv_weight_l2_diff",
    "gmv_weight_l1_diff",
    "tpr",
    "fpr",
    "precision",
    "recall",
    "f1",
    "mcc",
    "n_edges_detected",
    "n_edges_true",
    "coverage_95",
    "mean_interval_width",
    "mean_posterior_std_offdiag",
    "bimodality_coefficient_kappa",
    "shrinkage_wasserstein_vs_nuts",
]

HEADLINE_METRICS = [
    "steins_loss",
    "frobenius_loss_relative",
    "f1",
    "mcc",
    "coverage_95",
    "bimodality_coefficient_kappa",
]


def _finite_values(vals):
    """Keep finite float values, drop None/nan/inf."""
    out = []
    for v in vals:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _safe_mean(vals):
    vs = _finite_values(vals)
    return float(np.mean(vs)) if vs else None


def _safe_std(vals):
    vs = _finite_values(vals)
    return float(np.std(vs, ddof=1)) if len(vs) > 1 else (0.0 if vs else None)


def _walk_metrics(
    manifest_path: Path,
    results_root: Path,
    methods: List[str],
) -> Dict:
    """Return nested dict [config_id][method] = list of metric dicts."""
    with open(manifest_path) as f:
        configs = json.load(f)

    out: Dict = {}
    for cfg in configs:
        cid = int(cfg["config_id"])
        out[cid] = {m: [] for m in methods}
        for seed in range(int(cfg["n_seeds"])):
            for method in methods:
                mpath = (
                    results_root / cfg["dir_path"] / dir_name_seed(seed)
                    / method / "metrics.json"
                )
                if not mpath.exists():
                    continue
                try:
                    with open(mpath) as f:
                        m = json.load(f)
                except Exception:
                    continue
                out[cid][method].append(m)
    return {"configs": configs, "metrics": out}


def build_per_config_method(walk: Dict) -> List[dict]:
    """Return one dict per (config, method) with mean/std of every numeric metric."""
    configs = walk["configs"]
    metrics_tree = walk["metrics"]
    out: List[dict] = []
    for cfg in configs:
        cid = int(cfg["config_id"])
        for method in ALL_METHODS:
            rows = metrics_tree.get(cid, {}).get(method, [])
            success = [r for r in rows if r.get("status") == "success"]
            summary = {
                "config_id": cid,
                "method": method,
                "p": int(cfg["p"]),
                "T": int(cfg["T"]),
                "gamma": float(cfg["gamma"]),
                "graph": cfg["graph"],
                "sparsity": float(cfg["sparsity"]),
                "n_seeds_total": len(rows),
                "n_seeds_success": len(success),
                "n_seeds_failed": len(rows) - len(success),
            }
            # Elapsed from diagnostics-embedded fields if present in metrics.
            for field in NUMERIC_FIELDS:
                vals = [r.get(field) for r in success]
                summary[f"{field}_mean"] = _safe_mean(vals)
                summary[f"{field}_std"] = _safe_std(vals)
            out.append(summary)
    return out


def build_cross_method_table(per_cm: List[dict]) -> List[dict]:
    """Pivot: one dict per config with headline metrics for every method."""
    by_config: Dict[int, dict] = {}
    for row in per_cm:
        cid = row["config_id"]
        if cid not in by_config:
            by_config[cid] = {
                "config_id": cid,
                "p": row["p"],
                "T": row["T"],
                "gamma": row["gamma"],
                "graph": row["graph"],
                "sparsity": row["sparsity"],
                "methods": {},
            }
        entry = {}
        for metric in HEADLINE_METRICS:
            entry[f"{metric}_mean"] = row.get(f"{metric}_mean")
            entry[f"{metric}_std"] = row.get(f"{metric}_std")
        entry["n_seeds_success"] = row["n_seeds_success"]
        by_config[cid]["methods"][row["method"]] = entry
    return sorted(by_config.values(), key=lambda r: r["config_id"])


def build_loss_vs_gamma(per_cm: List[dict]) -> List[dict]:
    """Group by (p, graph, sparsity, method); each group holds a list of gamma points."""
    groups: Dict[tuple, list] = defaultdict(list)
    for row in per_cm:
        key = (row["p"], row["graph"], row["sparsity"], row["method"])
        groups[key].append(row)

    out = []
    for key, rows in sorted(groups.items()):
        p, graph, s, method = key
        rows_sorted = sorted(rows, key=lambda r: r["gamma"])
        out.append(
            {
                "p": p,
                "graph": graph,
                "sparsity": s,
                "method": method,
                "gammas": [r["gamma"] for r in rows_sorted],
                "T_values": [r["T"] for r in rows_sorted],
                "n_seeds_success": [r["n_seeds_success"] for r in rows_sorted],
                "steins_loss_mean": [r.get("steins_loss_mean") for r in rows_sorted],
                "steins_loss_std": [r.get("steins_loss_std") for r in rows_sorted],
                "frobenius_loss_relative_mean": [
                    r.get("frobenius_loss_relative_mean") for r in rows_sorted
                ],
                "f1_mean": [r.get("f1_mean") for r in rows_sorted],
                "f1_std": [r.get("f1_std") for r in rows_sorted],
                "mcc_mean": [r.get("mcc_mean") for r in rows_sorted],
                "bimodality_coefficient_kappa_mean": [
                    r.get("bimodality_coefficient_kappa_mean") for r in rows_sorted
                ],
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate WORK2 per-seed metrics into per-config summaries."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    args = parser.parse_args()

    walk = _walk_metrics(args.manifest, args.results_root, ALL_METHODS)
    per_cm = build_per_config_method(walk)
    cross = build_cross_method_table(per_cm)
    lvg = build_loss_vs_gamma(per_cm)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_config_method": args.output_dir / "per_config_method.json",
        "cross_method_table": args.output_dir / "cross_method_table.json",
        "loss_vs_gamma": args.output_dir / "loss_vs_gamma.json",
    }
    with open(paths["per_config_method"], "w") as f:
        json.dump(per_cm, f, indent=2)
    with open(paths["cross_method_table"], "w") as f:
        json.dump(cross, f, indent=2)
    with open(paths["loss_vs_gamma"], "w") as f:
        json.dump(lvg, f, indent=2)

    print(f"per_config_method.json: {len(per_cm)} rows  -> {paths['per_config_method']}")
    print(f"cross_method_table.json: {len(cross)} configs -> {paths['cross_method_table']}")
    print(f"loss_vs_gamma.json: {len(lvg)} curves -> {paths['loss_vs_gamma']}")


if __name__ == "__main__":
    main()
