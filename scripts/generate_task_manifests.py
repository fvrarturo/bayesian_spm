"""Generate task manifests for the three inference SLURM arrays.

The frequentist array does not need a task manifest: each task batches
all three methods × all 20 seeds for one config, so ``config_id`` alone
is enough (and the existing ``config_manifest.json`` is used directly).

The ADVI and NUTS arrays expand one row per ``(config_id, seed, method)``
triple, indexed by a contiguous ``task_id``.  The SLURM array uses
``$SLURM_ARRAY_TASK_ID`` to look up its assignment.

Optional ``--subset-tier N`` flag produces reduced manifests for the
progress-report slice:
    tier 1 = p=50 ER s=0.10 gamma in {0.67, 0.20}, seeds 0..4
    tier 2 = p=50, all gammas, ER + block, s in {0.10, 0.30}, seeds 0..4
    tier 3 = full grid (default)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG_MANIFEST = REPO_ROOT / "data" / "synthetic" / "configs" / "config_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "task_manifests"

ALL_ADVI_METHODS = ["advi_mf", "advi_fr"]
ALL_NUTS_METHODS = ["nuts"]
ALL_FREQ_METHODS = ["sample_cov", "ledoit_wolf", "glasso"]


# ----------------------------------------------------------------------
# Subset selectors
# ----------------------------------------------------------------------

def _subset_tier1(config: dict) -> bool:
    """Progress-report slice: p=50, ER, s=0.10, gamma in {0.67, 0.20}."""
    return (
        config["p"] == 50
        and config["graph"] == "erdos_renyi"
        and config["sparsity"] == 0.10
        and config["gamma"] in (0.67, 0.20)
    )


def _subset_tier2(config: dict) -> bool:
    """Core results: p=50, all gammas, both graphs, s in {0.10, 0.30}."""
    return (
        config["p"] == 50
        and config["sparsity"] in (0.10, 0.30)
    )


def _subset_tier3(config: dict) -> bool:
    """Full grid (all 84 configs)."""
    return True


TIER_FILTERS = {
    1: _subset_tier1,
    2: _subset_tier2,
    3: _subset_tier3,
}


def _seeds_for_tier(tier: int, n_seeds_total: int) -> List[int]:
    """How many seeds to expand per config at each tier."""
    if tier == 1:
        return list(range(5))
    if tier == 2:
        return list(range(5))
    return list(range(n_seeds_total))


# ----------------------------------------------------------------------
# Core expansion
# ----------------------------------------------------------------------

def _expand(
    configs: List[dict],
    methods: Iterable[str],
    tier: int,
) -> List[dict]:
    tasks: List[dict] = []
    filt = TIER_FILTERS[tier]
    task_id = 0
    for cfg in configs:
        if not filt(cfg):
            continue
        seeds = _seeds_for_tier(tier, int(cfg["n_seeds"]))
        for seed in seeds:
            for method in methods:
                tasks.append({
                    "task_id": task_id,
                    "config_id": int(cfg["config_id"]),
                    "seed": int(seed),
                    "method": method,
                    "p": int(cfg["p"]),
                    "T": int(cfg["T"]),
                    "gamma": float(cfg["gamma"]),
                    "graph": cfg["graph"],
                    "sparsity": float(cfg["sparsity"]),
                    "dir_path": cfg["dir_path"],
                })
                task_id += 1
    return tasks


def _freq_config_list(configs: List[dict], tier: int) -> List[dict]:
    """Task list for the frequentist array: one entry per config (no seed expansion)."""
    filt = TIER_FILTERS[tier]
    out = []
    task_id = 0
    for cfg in configs:
        if not filt(cfg):
            continue
        out.append({
            "task_id": task_id,
            "config_id": int(cfg["config_id"]),
            "p": int(cfg["p"]),
            "T": int(cfg["T"]),
            "gamma": float(cfg["gamma"]),
            "graph": cfg["graph"],
            "sparsity": float(cfg["sparsity"]),
            "dir_path": cfg["dir_path"],
            "n_seeds": int(cfg["n_seeds"]) if tier == 3 else len(_seeds_for_tier(tier, int(cfg["n_seeds"]))),
        })
        task_id += 1
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM task manifests for WORK2 inference arrays."
    )
    parser.add_argument("--config-manifest", type=Path, default=DEFAULT_CONFIG_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tier", type=int, default=3, choices=[1, 2, 3],
        help="Subset selector: 1=progress-report slice, 2=core p=50, 3=full grid.",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Optional suffix for output files (e.g. '_tier1').",
    )
    args = parser.parse_args()

    with open(args.config_manifest) as f:
        configs = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix or (f"_tier{args.tier}" if args.tier != 3 else "")

    # --- Frequentist manifest (list of configs) ---
    freq_tasks = _freq_config_list(configs, args.tier)
    freq_path = args.output_dir / f"freq{suffix}.json"
    with open(freq_path, "w") as f:
        json.dump(freq_tasks, f, indent=2)
    print(f"freq:  {len(freq_tasks):>5} tasks   -> {freq_path}")

    # --- ADVI manifest (task per config × seed × method) ---
    advi_tasks = _expand(configs, ALL_ADVI_METHODS, args.tier)
    advi_path = args.output_dir / f"advi{suffix}.json"
    with open(advi_path, "w") as f:
        json.dump(advi_tasks, f, indent=2)
    print(f"advi:  {len(advi_tasks):>5} tasks   -> {advi_path}")

    # --- NUTS manifest ---
    nuts_tasks = _expand(configs, ALL_NUTS_METHODS, args.tier)
    nuts_path = args.output_dir / f"nuts{suffix}.json"
    with open(nuts_path, "w") as f:
        json.dump(nuts_tasks, f, indent=2)
    print(f"nuts:  {len(nuts_tasks):>5} tasks   -> {nuts_path}")

    print()
    print(f"Tier {args.tier}: Use --array=0-{len(freq_tasks)-1} for freq, "
          f"--array=0-{len(advi_tasks)-1} for advi, --array=0-{len(nuts_tasks)-1} for nuts.")


if __name__ == "__main__":
    main()
