"""Generate synthetic (Omega, Sigma, Y) triples for the experiment grid.

This script is designed to be driven by a SLURM array where each task
processes all seeds for a single ``config_id``.  It also supports
``--all`` for local smoke tests.

Typical usage
-------------
Single config (SLURM array task):

    python scripts/generate_synthetic_data.py \
        --config-id $SLURM_ARRAY_TASK_ID \
        --manifest data/synthetic/configs/config_manifest.json \
        --output-dir data/synthetic

All configs, locally:

    python scripts/generate_synthetic_data.py --all \
        --manifest data/synthetic/configs/config_manifest.json \
        --output-dir data/synthetic

One seed for a specific config (debugging):

    python scripts/generate_synthetic_data.py --config-id 7 --seeds 0

Override the seed count from the manifest (e.g. for a 1-seed smoke test):

    python scripts/generate_synthetic_data.py --all --n-seeds 1

Design notes
------------
- Every simulation is a pure function of ``(config_id, seed)``.  The
  graph and the data are drawn from independent RNG streams seeded by
  ``graph_seed = seed`` and ``data_seed = seed + 10000``.
- Validation failures are LOGGED but do not crash the job.  The
  corresponding seed is simply skipped (no corrupted files written).
- Each seed directory is atomic: we write all four files or none.
- Exit code is non-zero if any seed failed, so SLURM can flag the task.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Allow running as "python scripts/<name>.py" from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.configs import dir_name_seed  # noqa: E402
from src.utils.matrix_utils import (  # noqa: E402
    sample_data_from_omega,
    sparse_omega_block_diagonal,
    sparse_omega_erdos_renyi,
)
from src.utils.validation import (  # noqa: E402
    validate_data,
    validate_omega,
    validate_sigma,
)


DATA_SEED_OFFSET = 10_000
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "synthetic" / "configs" / "config_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "synthetic"


logger = logging.getLogger("generate_synthetic_data")


# ----------------------------------------------------------------------
# Single-seed generation
# ----------------------------------------------------------------------

def _generate_omega(config: dict, graph_seed: int):
    """Dispatch to the appropriate graph generator."""
    p = config["p"]
    signal_range = tuple(config["signal_range"])
    sparsity = config["sparsity"]
    graph = config["graph"]

    if graph == "erdos_renyi":
        return sparse_omega_erdos_renyi(
            p=p,
            sparsity=sparsity,
            signal_range=signal_range,
            seed=graph_seed,
        )
    if graph == "block_diagonal":
        n_blocks = config["n_blocks"]
        if n_blocks is None:
            raise ValueError(f"block_diagonal config missing n_blocks: {config}")
        return sparse_omega_block_diagonal(
            p=p,
            n_blocks=n_blocks,
            intra_sparsity=sparsity,
            signal_range=signal_range,
            seed=graph_seed,
        )
    raise ValueError(f"Unknown graph type: {graph!r}")


def _compute_oracle_portfolio(Omega: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return the oracle GMV weights and portfolio variance.

    Weights: w* = Omega 1 / (1' Omega 1)
    Variance: sigma^2 = 1 / (1' Omega 1)
    """
    p = Omega.shape[0]
    ones = np.ones(p)
    numerator = Omega @ ones
    denom = float(ones @ numerator)
    weights = numerator / denom
    variance = 1.0 / denom
    return weights, variance


def _write_atomic(target_dir: Path, files: dict) -> None:
    """Write a dict of {filename: contents} to target_dir atomically.

    Files are first written to a sibling ``.tmp`` directory and then
    moved into place.  If any write fails, the partial .tmp directory
    is left behind for inspection but the target directory remains
    untouched.
    """
    tmp_dir = target_dir.with_suffix(target_dir.suffix + ".tmp")
    if tmp_dir.exists():
        _rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in files.items():
        path = tmp_dir / name
        if isinstance(payload, np.ndarray):
            np.save(path, payload)
        elif isinstance(payload, (dict, list)):
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        else:
            raise TypeError(f"Unsupported payload type for {name}: {type(payload)}")

    # Move into place: replace target_dir with tmp_dir.
    if target_dir.exists():
        _rmtree(target_dir)
    tmp_dir.rename(target_dir)


def _rmtree(path: Path) -> None:
    """Recursively delete a directory tree."""
    import shutil

    shutil.rmtree(path)


def generate_single_config(
    config: dict,
    seed: int,
    output_base_dir: Path,
) -> Tuple[bool, dict]:
    """Generate, validate, and save one (config, seed) pair.

    Returns
    -------
    (success, info)
        ``success`` is True iff all data was generated and saved.
        ``info`` is a dict with the metadata (on success) or an
        ``error`` key (on failure).
    """
    p = int(config["p"])
    T = int(config["T"])
    graph_seed = int(seed)
    data_seed = int(seed) + DATA_SEED_OFFSET

    # --- 1. Omega ---
    try:
        Omega, edge_set, gen_diag = _generate_omega(config, graph_seed)
    except Exception as exc:  # pragma: no cover - defensive
        return False, {"error": f"graph generation raised: {exc!r}"}

    shift = gen_diag.get("diagonal_shift", 0.0)
    ok, omega_diag = validate_omega(Omega, edge_set, p, diagonal_shift=shift)
    if not ok:
        return False, {"stage": "validate_omega", **omega_diag}

    # --- 2. Sigma ---
    try:
        Sigma = np.linalg.inv(Omega)
    except np.linalg.LinAlgError as exc:
        return False, {"stage": "invert_omega", "error": str(exc)}

    ok, sigma_diag = validate_sigma(Sigma, Omega, p)
    if not ok:
        return False, {"stage": "validate_sigma", **sigma_diag}

    # --- 3. Y ---
    Y = sample_data_from_omega(Omega, T=T, seed=data_seed)
    ok, y_diag = validate_data(Y, T=T, p=p, Sigma=Sigma)
    if not ok:
        return False, {"stage": "validate_data", **y_diag}

    # --- 4. Metadata ---
    eigs = np.linalg.eigvalsh(Omega)
    _, oracle_variance = _compute_oracle_portfolio(Omega)
    n_possible_edges = p * (p - 1) // 2
    realized_sparsity = len(edge_set) / n_possible_edges if n_possible_edges > 0 else 0.0

    warnings: List[str] = []
    warnings.extend(omega_diag.get("warnings", []))
    warnings.extend(sigma_diag.get("warnings", []))
    warnings.extend(y_diag.get("warnings", []))

    metadata = {
        "config_id": int(config["config_id"]),
        "p": p,
        "T": T,
        "gamma": float(config["gamma"]),
        "graph": config["graph"],
        "sparsity": float(config["sparsity"]),
        "n_blocks": config.get("n_blocks"),
        "signal_range": list(config["signal_range"]),
        "seed": int(seed),
        "graph_seed": int(graph_seed),
        "data_seed": int(data_seed),
        "n_edges": int(len(edge_set)),
        "n_possible_edges": int(n_possible_edges),
        "realized_sparsity": float(realized_sparsity),
        "condition_number": float(eigs.max() / eigs.min()),
        "min_eigenvalue": float(eigs.min()),
        "max_eigenvalue": float(eigs.max()),
        "trace_omega": float(np.trace(Omega)),
        "oracle_portfolio_variance": float(oracle_variance),
        "diagonal_shift_applied": float(shift),
        "min_eigenvalue_pre_shift": float(
            gen_diag.get("min_eigenvalue_pre_shift", float("nan"))
        ),
        "max_mean_zscore": float(y_diag.get("max_mean_zscore", float("nan"))),
        "max_abs_mean": float(y_diag.get("max_abs_mean", float("nan"))),
        "sample_cov_rank": int(y_diag.get("sample_cov_rank", -1)),
        "warnings": warnings,
        "edge_set": [list(e) for e in sorted(edge_set)],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # --- 5. Save atomically ---
    seed_dir = output_base_dir / config["dir_path"] / dir_name_seed(seed)
    try:
        _write_atomic(
            seed_dir,
            {
                "omega_true.npy": Omega.astype(np.float64),
                "sigma_true.npy": Sigma.astype(np.float64),
                "Y.npy": Y.astype(np.float64),
                "metadata.json": metadata,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        return False, {"stage": "write", "error": repr(exc)}

    return True, metadata


# ----------------------------------------------------------------------
# Config-level and batch driver
# ----------------------------------------------------------------------

def generate_config_all_seeds(
    config: dict,
    output_base_dir: Path,
    n_seeds: Optional[int] = None,
    explicit_seeds: Optional[List[int]] = None,
) -> dict:
    """Generate all seeds for a single configuration.

    Returns a summary dict with counts and lists of failures.
    """
    if explicit_seeds is not None:
        seeds = list(explicit_seeds)
    else:
        count = n_seeds if n_seeds is not None else int(config["n_seeds"])
        seeds = list(range(count))

    summary = {
        "config_id": int(config["config_id"]),
        "graph": config["graph"],
        "p": int(config["p"]),
        "gamma": float(config["gamma"]),
        "sparsity": float(config["sparsity"]),
        "T": int(config["T"]),
        "dir_path": config["dir_path"],
        "n_seeds_requested": len(seeds),
        "n_seeds_success": 0,
        "n_seeds_failed": 0,
        "failures": [],
        "warnings_per_seed": {},
    }

    for seed in seeds:
        t0 = time.time()
        ok, info = generate_single_config(config, seed, output_base_dir)
        elapsed = time.time() - t0

        if ok:
            summary["n_seeds_success"] += 1
            if info.get("warnings"):
                summary["warnings_per_seed"][seed] = info["warnings"]
            logger.info(
                "  ok  cfg=%d seed=%d  n_edges=%d  kappa=%.1f  shift=%.3f  %.2fs",
                config["config_id"],
                seed,
                info["n_edges"],
                info["condition_number"],
                info["diagonal_shift_applied"],
                elapsed,
            )
        else:
            summary["n_seeds_failed"] += 1
            summary["failures"].append({"seed": seed, **info})
            logger.error(
                "  FAIL cfg=%d seed=%d  stage=%s  error=%s",
                config["config_id"],
                seed,
                info.get("stage", "?"),
                info.get("error", "?"),
            )

    return summary


def generate_all_configs(
    manifest_path: Path,
    output_base_dir: Path,
    n_seeds: Optional[int] = None,
    config_ids: Optional[List[int]] = None,
    explicit_seeds: Optional[List[int]] = None,
) -> dict:
    """Generate many configs from a manifest file.

    Returns a top-level summary dict across all processed configs.
    """
    with open(manifest_path) as f:
        all_configs = json.load(f)

    if config_ids is not None:
        selected = [c for c in all_configs if c["config_id"] in set(config_ids)]
    else:
        selected = all_configs

    logger.info(
        "Generating %d configs from %s into %s",
        len(selected),
        manifest_path,
        output_base_dir,
    )

    per_config = []
    total_success = 0
    total_failed = 0
    start = time.time()

    for config in selected:
        logger.info(
            "Config %d: p=%d gamma=%.2f graph=%s s=%.2f T=%d",
            config["config_id"],
            config["p"],
            config["gamma"],
            config["graph"],
            config["sparsity"],
            config["T"],
        )
        result = generate_config_all_seeds(
            config,
            output_base_dir,
            n_seeds=n_seeds,
            explicit_seeds=explicit_seeds,
        )
        per_config.append(result)
        total_success += result["n_seeds_success"]
        total_failed += result["n_seeds_failed"]

    elapsed = time.time() - start

    summary = {
        "n_configs": len(selected),
        "total_success": total_success,
        "total_failed": total_failed,
        "elapsed_seconds": elapsed,
        "per_config": per_config,
        "manifest_path": str(manifest_path),
        "output_base_dir": str(output_base_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Done: %d configs, %d successes, %d failures in %.1fs",
        summary["n_configs"],
        summary["total_success"],
        summary["total_failed"],
        summary["elapsed_seconds"],
    )
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_seed_list(text: str) -> List[int]:
    """Parse a comma-separated seed list like '0,1,5,9'."""
    return [int(s) for s in text.split(",") if s.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic (Omega, Sigma, Y) data for the experiment grid."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to config_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory under which config trees are written.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--config-id",
        type=int,
        help="Process a single config by ID (SLURM array mode).",
    )
    mode.add_argument(
        "--config-ids",
        type=str,
        help="Comma-separated list of config IDs to process.",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Process every config in the manifest.",
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Override n_seeds from the manifest (useful for smoke tests).",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default=None,
        help="Comma-separated explicit seeds to process (e.g. '0,1,5').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="If provided, write a JSON summary of the run to this path.",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_level)

    if args.config_id is not None:
        config_ids = [args.config_id]
    elif args.config_ids is not None:
        config_ids = _parse_seed_list(args.config_ids)
    else:
        config_ids = None  # all

    summary = generate_all_configs(
        manifest_path=args.manifest,
        output_base_dir=args.output_dir,
        n_seeds=args.n_seeds,
        config_ids=config_ids,
        explicit_seeds=args.seeds,
    )

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary written to %s", args.summary_path)

    # Non-zero exit if anything failed, so SLURM marks the task as FAILED.
    return 0 if summary["total_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
