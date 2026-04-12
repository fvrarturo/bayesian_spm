"""Post-run audit for WORK2 inference results.

Walks ``results/synthetic/`` and, for every (config_id, seed, method)
expected by the config manifest, verifies that a minimal set of files
exists and is parseable:

- ``diagnostics.json`` — always expected
- ``metrics.json`` — always expected (may contain null values for
  failed/timeout runs)
- ``omega_hat.npy`` — expected iff ``status == "success"``

Reports per-method counts of statuses and flags any missing or
corrupted outputs.  Exit code 0 iff every expected task has a
parseable diagnostics.json and metrics.json.

Usage
-----
    python scripts/audit_results.py                         # all methods
    python scripts/audit_results.py --method nuts           # one method
    python scripts/audit_results.py --strict                # also load omega_hat
    python scripts/audit_results.py --report logs/audit_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.configs import dir_name_seed  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "data" / "synthetic" / "configs" / "config_manifest.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "synthetic"

ALL_METHODS = ["nuts", "advi_mf", "advi_fr", "glasso", "ledoit_wolf", "sample_cov"]


def _audit_one(
    method_dir: Path,
    method: str,
    config: dict,
    seed: int,
    strict: bool,
) -> dict:
    entry = {
        "config_id": config["config_id"],
        "seed": seed,
        "method": method,
        "path": str(method_dir),
        "status": None,
        "inference_status": None,
        "detail": None,
    }

    if not method_dir.exists():
        entry["status"] = "missing_dir"
        return entry

    diag_path = method_dir / "diagnostics.json"
    if not diag_path.exists():
        entry["status"] = "missing_diagnostics"
        return entry

    try:
        with open(diag_path) as f:
            diag = json.load(f)
    except Exception as e:
        entry["status"] = "corrupted_diagnostics"
        entry["detail"] = repr(e)
        return entry

    entry["inference_status"] = diag.get("status", "unknown")
    entry["elapsed_seconds"] = diag.get("elapsed_seconds")

    metrics_path = method_dir / "metrics.json"
    if not metrics_path.exists():
        entry["status"] = "missing_metrics"
        return entry

    try:
        with open(metrics_path) as f:
            _ = json.load(f)
    except Exception as e:
        entry["status"] = "corrupted_metrics"
        entry["detail"] = repr(e)
        return entry

    omega_hat_path = method_dir / "omega_hat.npy"
    expect_estimate = diag.get("status") == "success"
    if expect_estimate and not omega_hat_path.exists():
        entry["status"] = "missing_omega_hat"
        return entry

    if strict and omega_hat_path.exists():
        try:
            arr = np.load(omega_hat_path)
            p = int(config["p"])
            if arr.shape != (p, p):
                entry["status"] = "corrupted_omega_hat"
                entry["detail"] = f"shape {arr.shape} != ({p}, {p})"
                return entry
            if not np.isfinite(arr).all():
                entry["status"] = "corrupted_omega_hat"
                entry["detail"] = "NaN/Inf in omega_hat"
                return entry
        except Exception as e:
            entry["status"] = "corrupted_omega_hat"
            entry["detail"] = repr(e)
            return entry

    entry["status"] = "ok"
    return entry


def audit(
    manifest_path: Path,
    results_root: Path,
    methods: List[str],
    strict: bool = False,
) -> dict:
    with open(manifest_path) as f:
        configs = json.load(f)

    per_method_status: dict = defaultdict(lambda: Counter())
    per_method_inf_status: dict = defaultdict(lambda: Counter())
    per_method_elapsed: dict = defaultdict(list)
    all_entries: List[dict] = []
    failures: List[dict] = []

    for config in configs:
        for seed in range(int(config["n_seeds"])):
            for method in methods:
                method_dir = (
                    results_root
                    / config["dir_path"]
                    / dir_name_seed(seed)
                    / method
                )
                entry = _audit_one(method_dir, method, config, seed, strict)
                all_entries.append(entry)
                status = entry["status"]
                per_method_status[method][status] += 1
                if entry.get("inference_status") is not None:
                    per_method_inf_status[method][entry["inference_status"]] += 1
                if entry.get("elapsed_seconds") is not None:
                    per_method_elapsed[method].append(entry["elapsed_seconds"])
                if status != "ok":
                    failures.append(entry)

    # Aggregate summary
    per_method_summary = {}
    for method in methods:
        elapsed = per_method_elapsed[method]
        per_method_summary[method] = {
            "audit_status": dict(per_method_status[method]),
            "inference_status": dict(per_method_inf_status[method]),
            "n_tasks": sum(per_method_status[method].values()),
            "mean_elapsed": float(np.mean(elapsed)) if elapsed else None,
            "median_elapsed": float(np.median(elapsed)) if elapsed else None,
            "max_elapsed": float(np.max(elapsed)) if elapsed else None,
        }

    summary = {
        "manifest_path": str(manifest_path),
        "results_root": str(results_root),
        "strict": strict,
        "methods": methods,
        "n_configs": len(configs),
        "n_tasks_total": len(all_entries),
        "n_failures": len(failures),
        "per_method": per_method_summary,
        "failures": failures[:200],
        "failures_truncated": len(failures) > 200,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit WORK2 inference results against the config manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Audit a single method instead of all six.",
    )
    parser.add_argument("--strict", action="store_true",
                        help="Also load each omega_hat.npy and verify shape/finiteness.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Write the full audit as JSON to this path.")
    args = parser.parse_args()

    methods = [args.method] if args.method else list(ALL_METHODS)
    summary = audit(args.manifest, args.results_root, methods, strict=args.strict)

    print(f"Audit: manifest={args.manifest}  results_root={args.results_root}")
    print(f"  methods:     {summary['methods']}")
    print(f"  configs:     {summary['n_configs']}")
    print(f"  tasks total: {summary['n_tasks_total']}")
    print(f"  failures:    {summary['n_failures']}")
    print()
    print(f"{'method':<14}{'n':>6}{'ok':>6}{'missing_dir':>14}"
          f"{'missing_diag':>14}{'missing_met':>14}{'missing_est':>14}")
    for m, ps in summary["per_method"].items():
        s = ps["audit_status"]
        print(
            f"{m:<14}{ps['n_tasks']:>6}"
            f"{s.get('ok', 0):>6}"
            f"{s.get('missing_dir', 0):>14}"
            f"{s.get('missing_diagnostics', 0):>14}"
            f"{s.get('missing_metrics', 0):>14}"
            f"{s.get('missing_omega_hat', 0):>14}"
        )
    print()
    print("Inference status breakdown (from diagnostics.json):")
    print(f"{'method':<14}{'success':>10}{'timeout':>10}{'failed':>10}"
          f"{'singular':>10}{'other':>10}")
    for m, ps in summary["per_method"].items():
        inf = ps["inference_status"]
        other = sum(v for k, v in inf.items() if k not in ("success", "timeout", "failed", "singular"))
        print(
            f"{m:<14}{inf.get('success', 0):>10}{inf.get('timeout', 0):>10}"
            f"{inf.get('failed', 0):>10}{inf.get('singular', 0):>10}{other:>10}"
        )
    print()
    print("Mean elapsed per method (seconds):")
    for m, ps in summary["per_method"].items():
        if ps["mean_elapsed"] is not None:
            print(f"  {m:<14}  mean={ps['mean_elapsed']:>9.1f}  "
                  f"median={ps['median_elapsed']:>9.1f}  max={ps['max_elapsed']:>9.1f}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull report written to {args.report}")

    return 0 if summary["n_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
