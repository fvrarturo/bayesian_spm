"""Audit a generated synthetic data tree against the manifest.

Walks ``data/synthetic/`` and, for every config listed in the manifest,
verifies that each expected ``seed_NN/`` directory exists and contains
the four expected files: ``omega_true.npy``, ``sigma_true.npy``,
``Y.npy``, ``metadata.json``.

Reports counts for:
- configs fully present
- seeds fully present
- missing seed directories
- partially-present seed directories (missing files)
- corrupted files (zero bytes or unloadable)
- metadata <-> manifest inconsistencies (wrong p, T, config_id, etc.)

Exit code is 0 iff everything is present and consistent.

Usage
-----
    python scripts/audit_synthetic_data.py
    python scripts/audit_synthetic_data.py --strict       # also verify arrays load
    python scripts/audit_synthetic_data.py --report out.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.configs import dir_name_seed  # noqa: E402


DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "synthetic" / "configs" / "config_manifest.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "synthetic"

EXPECTED_FILES = ("omega_true.npy", "sigma_true.npy", "Y.npy", "metadata.json")


def _audit_seed_dir(
    seed_dir: Path,
    config: dict,
    seed: int,
    strict: bool,
) -> Dict[str, Optional[str]]:
    """Audit a single seed directory.

    Returns a dict with one of the following ``status`` values:
    ``"ok"``, ``"missing_dir"``, ``"missing_files"``, ``"corrupted"``,
    ``"metadata_mismatch"``.
    """
    result: Dict[str, Optional[str]] = {
        "config_id": config["config_id"],
        "seed": seed,
        "path": str(seed_dir),
        "status": None,
        "detail": None,
    }

    if not seed_dir.exists() or not seed_dir.is_dir():
        result["status"] = "missing_dir"
        return result

    missing = [f for f in EXPECTED_FILES if not (seed_dir / f).exists()]
    if missing:
        result["status"] = "missing_files"
        result["detail"] = ", ".join(missing)
        return result

    # Zero-byte detection
    empties = [f for f in EXPECTED_FILES if (seed_dir / f).stat().st_size == 0]
    if empties:
        result["status"] = "corrupted"
        result["detail"] = f"zero-byte files: {', '.join(empties)}"
        return result

    # Metadata consistency
    try:
        with open(seed_dir / "metadata.json") as f:
            metadata = json.load(f)
    except Exception as exc:
        result["status"] = "corrupted"
        result["detail"] = f"metadata.json unreadable: {exc!r}"
        return result

    expected_pairs = {
        "config_id": config["config_id"],
        "p": config["p"],
        "T": config["T"],
        "graph": config["graph"],
        "sparsity": config["sparsity"],
        "seed": seed,
    }
    mismatches = {
        k: (metadata.get(k), v)
        for k, v in expected_pairs.items()
        if metadata.get(k) != v
    }
    if mismatches:
        result["status"] = "metadata_mismatch"
        result["detail"] = json.dumps(mismatches)
        return result

    # Optional strict: actually try to load arrays and check shapes
    if strict:
        try:
            Omega = np.load(seed_dir / "omega_true.npy")
            Sigma = np.load(seed_dir / "sigma_true.npy")
            Y = np.load(seed_dir / "Y.npy")
        except Exception as exc:
            result["status"] = "corrupted"
            result["detail"] = f"array load failed: {exc!r}"
            return result

        p = config["p"]
        T = config["T"]
        expected_shapes = {
            "omega_true.npy": (Omega, (p, p)),
            "sigma_true.npy": (Sigma, (p, p)),
            "Y.npy": (Y, (T, p)),
        }
        bad_shapes = {
            name: arr.shape
            for name, (arr, want) in expected_shapes.items()
            if arr.shape != want
        }
        if bad_shapes:
            result["status"] = "corrupted"
            result["detail"] = f"wrong shapes: {bad_shapes}"
            return result

        if not np.isfinite(Omega).all() or not np.isfinite(Sigma).all() or not np.isfinite(Y).all():
            result["status"] = "corrupted"
            result["detail"] = "NaN/Inf in saved arrays"
            return result

    result["status"] = "ok"
    return result


def audit(
    manifest_path: Path,
    data_root: Path,
    strict: bool = False,
) -> dict:
    """Run the full audit and return a summary dict."""
    with open(manifest_path) as f:
        configs = json.load(f)

    per_seed: List[dict] = []
    status_counts: Dict[str, int] = {
        "ok": 0,
        "missing_dir": 0,
        "missing_files": 0,
        "corrupted": 0,
        "metadata_mismatch": 0,
    }

    configs_fully_ok = 0
    for config in configs:
        seeds_ok = 0
        for seed in range(int(config["n_seeds"])):
            seed_dir = data_root / config["dir_path"] / dir_name_seed(seed)
            entry = _audit_seed_dir(seed_dir, config, seed, strict=strict)
            per_seed.append(entry)
            status = entry["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "ok":
                seeds_ok += 1
        if seeds_ok == int(config["n_seeds"]):
            configs_fully_ok += 1

    total = len(per_seed)
    failures = [e for e in per_seed if e["status"] != "ok"]

    summary = {
        "manifest_path": str(manifest_path),
        "data_root": str(data_root),
        "strict": strict,
        "n_configs": len(configs),
        "configs_fully_ok": configs_fully_ok,
        "n_seeds_total": total,
        "status_counts": status_counts,
        "n_failures": len(failures),
        "failures": failures[:200],  # cap to keep report small
        "failures_truncated": len(failures) > 200,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a generated synthetic data tree."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also load each array and verify its shape and finiteness.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="If provided, write the full JSON audit summary to this path.",
    )
    args = parser.parse_args()

    summary = audit(args.manifest, args.data_root, strict=args.strict)

    print(f"Audit: manifest={args.manifest}  data_root={args.data_root}")
    print(f"  configs:               {summary['n_configs']}")
    print(f"  configs fully present: {summary['configs_fully_ok']}")
    print(f"  seed dirs expected:    {summary['n_seeds_total']}")
    for status, count in summary["status_counts"].items():
        print(f"    {status:<20} {count}")
    if summary["n_failures"]:
        print("\nFirst few failures:")
        for entry in summary["failures"][:10]:
            print(
                f"  cfg={entry['config_id']:<3} seed={entry['seed']:<2}  "
                f"{entry['status']:<18}  {entry['detail'] or ''}  {entry['path']}"
            )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull report written to {args.report}")

    return 0 if summary["n_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
