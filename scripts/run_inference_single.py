"""CLI driver for single / batch inference tasks.

Three invocation modes:

1. **Task-manifest mode** (used by SLURM arrays for NUTS and ADVI):

       python scripts/run_inference_single.py \
           --task-id $SLURM_ARRAY_TASK_ID \
           --task-manifest results/task_manifests/nuts.json

   Looks up ``(config_id, seed, method)`` by task id and runs exactly one.

2. **Direct single mode** (used for smoke tests and debugging):

       python scripts/run_inference_single.py \
           --config-id 7 --seed 0 --method nuts

3. **Batch mode** (used by the frequentist SLURM array to pack many
   fast methods into one task):

       python scripts/run_inference_single.py \
           --config-id 7 --methods sample_cov,ledoit_wolf,glasso \
           --seeds all

Exit code: 0 iff every requested run produced ``status="success"``
(or ``status="singular"`` for ``sample_cov``, which is a legitimate
outcome at high p/T).  1 otherwise.
"""

from __future__ import annotations

# ============================================================
# IMPORTANT: these environment settings must be applied BEFORE any
# import that triggers JAX initialization.  Once JAX has allocated
# its runtime, changing device count is a no-op.
# ============================================================
import os  # noqa: E402

# Use 4 virtual host devices so NUTS can run its 4 chains in parallel
# on a single CPU node.  Users can override via NUM_DEVICES before
# invocation (e.g. on GPU nodes).
_NUM_DEVICES = os.environ.get("NUM_DEVICES", "4")
_existing = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _existing:
    os.environ["XLA_FLAGS"] = (
        _existing + f" --xla_force_host_platform_device_count={_NUM_DEVICES}"
    ).strip()

# Keep JAX / MKL from contending with SLURM's single-CPU allocation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluate_single import evaluate  # noqa: E402
from src.inference.run_single import (  # noqa: E402
    DEFAULT_ADVI_TIMEOUT_SECONDS,
    DEFAULT_FREQ_TIMEOUT_SECONDS,
    DEFAULT_NUTS_TIMEOUT_SECONDS,
    run_inference,
)
from src.utils.configs import dir_name_seed  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "synthetic"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "synthetic"
DEFAULT_CONFIG_MANIFEST = REPO_ROOT / "data" / "synthetic" / "configs" / "config_manifest.json"

# Statuses that we treat as "not a failure" for purposes of the process exit code.
# 'singular' = sample_cov at T<=p, which is an expected outcome, not a bug.
_OK_STATUSES = {"success", "singular"}

logger = logging.getLogger("run_inference_single")


# ======================================================================
# Helpers
# ======================================================================

def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _build_dirs(config: dict, seed: int, method: str,
                data_root: Path, results_root: Path):
    data_dir = data_root / config["dir_path"] / dir_name_seed(seed)
    output_dir = results_root / config["dir_path"] / dir_name_seed(seed) / method
    return data_dir, output_dir


def _default_timeout_for_method(method: str) -> int:
    if method == "nuts":
        return DEFAULT_NUTS_TIMEOUT_SECONDS
    if method in ("advi_mf", "advi_fr"):
        return DEFAULT_ADVI_TIMEOUT_SECONDS
    return DEFAULT_FREQ_TIMEOUT_SECONDS


def _run_one(
    config: dict,
    seed: int,
    method: str,
    data_root: Path,
    results_root: Path,
    timeout_seconds: Optional[int],
    skip_existing: bool,
    skip_eval: bool,
) -> dict:
    """Run one (config, seed, method) end-to-end: inference + evaluation."""
    data_dir, output_dir = _build_dirs(
        config, seed, method, data_root, results_root
    )

    if skip_existing and (output_dir / "metrics.json").exists():
        logger.info(
            "  SKIP cfg=%d seed=%d method=%s (metrics.json already exists)",
            config["config_id"], seed, method,
        )
        return {"status": "skipped"}

    if not data_dir.exists():
        logger.error("  FAIL cfg=%d seed=%d method=%s  data dir missing: %s",
                     config["config_id"], seed, method, data_dir)
        return {"status": "missing_data"}

    t0 = time.time()
    if timeout_seconds is None:
        timeout_seconds = _default_timeout_for_method(method)

    diag = run_inference(
        method=method,
        data_dir=data_dir,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
    )

    # Always run the evaluator so a metrics.json exists (with nulls on failure).
    if not skip_eval:
        try:
            evaluate(method, data_dir, output_dir)
        except Exception as e:
            logger.error(
                "  evaluate() raised for cfg=%d seed=%d method=%s: %r",
                config["config_id"], seed, method, e,
            )

    elapsed = time.time() - t0
    status = diag.get("status", "unknown")

    if status == "success":
        logger.info(
            "  ok   cfg=%d seed=%d method=%s  %.1fs",
            config["config_id"], seed, method, elapsed,
        )
    elif status == "singular":
        logger.info(
            "  SING cfg=%d seed=%d method=%s  (expected at T<=p)",
            config["config_id"], seed, method,
        )
    elif status == "timeout":
        logger.warning(
            "  TOUT cfg=%d seed=%d method=%s  %.1fs",
            config["config_id"], seed, method, elapsed,
        )
    else:
        logger.error(
            "  FAIL cfg=%d seed=%d method=%s  %.1fs  %s",
            config["config_id"], seed, method, elapsed,
            diag.get("error", "unknown error"),
        )

    return diag


# ======================================================================
# Argument parsing
# ======================================================================

def _parse_seed_list(text: Optional[str], n_seeds_total: int) -> List[int]:
    if text is None or text == "all":
        return list(range(n_seeds_total))
    return [int(s) for s in text.split(",") if s.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one or many inference tasks on WORK1 synthetic data."
    )

    # --- mode selection ---
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-id", type=int,
                      help="SLURM array task id.  Requires --task-manifest.")
    mode.add_argument("--config-id", type=int,
                      help="Run directly on the given config.")

    # --- task manifest mode args ---
    parser.add_argument("--task-manifest", type=Path,
                        help="Task manifest JSON file.")

    # --- direct / batch mode args ---
    parser.add_argument("--method", type=str,
                        help="Single method to run.")
    parser.add_argument(
        "--methods", type=str,
        help=(
            "Comma-separated list of methods.  If provided, overrides --method. "
            "Used by the frequentist batch SLURM array."
        ),
    )
    parser.add_argument("--seed", type=int,
                        help="Single seed (direct mode).")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list or 'all' (batch mode).")

    # --- shared args ---
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--config-manifest", type=Path, default=DEFAULT_CONFIG_MANIFEST)
    parser.add_argument(
        "--timeout-seconds", type=int, default=None,
        help="Override the per-method default timeout.  0 disables.",
    )
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tasks whose metrics.json already exists.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Run inference but not evaluation.")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ======================================================================
# Main entry
# ======================================================================

def main() -> int:
    args = parse_args()
    _configure_logging(args.log_level)

    configs = _load_json(args.config_manifest)
    configs_by_id = {c["config_id"]: c for c in configs}

    runs: List[tuple] = []  # (config, seed, method)

    # ---- Mode 1: task manifest ----
    if args.task_id is not None:
        if args.task_manifest is None:
            raise SystemExit("--task-id requires --task-manifest")
        tasks = _load_json(args.task_manifest)
        task = next((t for t in tasks if int(t["task_id"]) == int(args.task_id)), None)
        if task is None:
            raise SystemExit(f"task_id {args.task_id} not found in {args.task_manifest}")
        config = configs_by_id[int(task["config_id"])]
        runs.append((config, int(task["seed"]), task["method"]))

    # ---- Modes 2 & 3: direct or batch ----
    else:
        config = configs_by_id[int(args.config_id)]
        n_seeds = int(config["n_seeds"])

        if args.methods:
            methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        elif args.method:
            methods = [args.method]
        else:
            raise SystemExit("direct mode requires --method or --methods")

        if args.seed is not None:
            seeds = [int(args.seed)]
        else:
            seeds = _parse_seed_list(args.seeds, n_seeds)

        for method in methods:
            for seed in seeds:
                runs.append((config, seed, method))

    # ---- Execute ----
    logger.info("Running %d task(s)", len(runs))
    n_ok = n_skip = n_fail = 0
    for config, seed, method in runs:
        diag = _run_one(
            config=config,
            seed=seed,
            method=method,
            data_root=args.data_root,
            results_root=args.results_root,
            timeout_seconds=args.timeout_seconds,
            skip_existing=args.skip_existing,
            skip_eval=args.skip_eval,
        )
        status = diag.get("status", "unknown")
        if status == "skipped":
            n_skip += 1
        elif status in _OK_STATUSES:
            n_ok += 1
        else:
            n_fail += 1

    logger.info("Done: %d ok, %d skipped, %d failed", n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
