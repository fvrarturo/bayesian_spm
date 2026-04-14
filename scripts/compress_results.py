"""Convert legacy ``.npy`` sample arrays to compressed ``.npz``.

Walks ``results/synthetic/`` and converts the bulky posterior-sample
arrays in place:

    omega_samples.npy       -> omega_samples.npz        (float32, kept)
    kappa_samples.npy       -> kappa_samples.npz        (float32, downcast)
    lambda_samples.npy      -> lambda_samples.npz       (float32, downcast)
    omega_diag_samples.npy  -> omega_diag_samples.npz   (float32, downcast)
    tau_samples.npy         -> tau_samples.npz          (float32, downcast)
    elbo_trace.npy          -> elbo_trace.npz           (float32, kept)

Small files (``omega_hat.npy``, ``sigma_hat.npy``, ``offdiag_magnitudes.npy``)
are left alone — they're referenced by many scripts and the compression
win is negligible.

The float64→float32 downcast is safe: every array here is a stochastic
Monte Carlo draw, and MCMC/SVI noise (~1/sqrt(N) ≈ 1e-2) is five orders
of magnitude larger than float32 precision (~1e-7).

Usage
-----
    # dry run: report what would change
    python scripts/compress_results.py --dry-run

    # actual conversion (default: delete .npy after successful .npz write)
    python scripts/compress_results.py

    # keep the .npy files around (for a paranoid first pass)
    python scripts/compress_results.py --keep-npy

The conversion is **safe under interruption**: each file is written to
``<name>.npz.tmp`` first and atomically renamed, and the ``.npy`` is
deleted only after the ``.npz`` is on disk.  Re-running the script skips
files already converted.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import save_samples_compressed  # noqa: E402

# (name, target_dtype).  dtype=None means keep source dtype.
COMPRESS_SPECS = (
    ("omega_samples", np.float32),        # already float32 on disk
    ("kappa_samples", np.float32),        # was float64
    ("lambda_samples", np.float32),       # was float64
    ("omega_diag_samples", np.float32),   # was float64
    ("tau_samples", np.float32),          # was float64
    ("elbo_trace", np.float32),           # already float32 on disk
)


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _convert_one(
    npy_path: Path,
    dtype,
    *,
    delete_npy: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """Convert a single .npy file.  Returns (bytes_before, bytes_after)."""
    npz_path = npy_path.with_suffix(".npz")
    before = npy_path.stat().st_size

    if dry_run:
        # Estimate only: we know the original size; the compressed size
        # depends on the data.  Skip the compute.
        return before, 0

    # If .npz already exists, skip but still possibly delete leftover .npy.
    if npz_path.exists():
        if delete_npy:
            npy_path.unlink()
        return before, npz_path.stat().st_size

    arr = np.load(npy_path)
    # ``np.savez_compressed`` auto-appends ``.npz`` unless the filename
    # already ends in it, so the tmp name MUST end in ``.npz``.
    tmp = npz_path.with_suffix(".tmp.npz")
    save_samples_compressed(tmp, arr, dtype=dtype)
    tmp.replace(npz_path)
    after = npz_path.stat().st_size

    if delete_npy:
        npy_path.unlink()

    return before, after


def _iter_targets(root: Path) -> Iterable[tuple[Path, "np.dtype | None"]]:
    for name, dtype in COMPRESS_SPECS:
        yield from ((p, dtype) for p in root.rglob(f"{name}.npy"))


def _convert_worker(task):
    """Picklable wrapper around ``_convert_one`` for a multiprocessing Pool."""
    npy_path, dtype, delete_npy, dry_run = task
    try:
        before, after = _convert_one(
            npy_path, dtype,
            delete_npy=delete_npy,
            dry_run=dry_run,
        )
        return (npy_path, before, after, None)
    except Exception as e:
        return (npy_path, 0, 0, repr(e))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compress legacy .npy posterior-sample arrays to .npz.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "results" / "synthetic",
        help="Root directory to walk (default: results/synthetic).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be converted without writing anything.",
    )
    ap.add_argument(
        "--keep-npy",
        action="store_true",
        help="Keep the original .npy alongside the new .npz.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Only print per-file entries on error; show totals at the end.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, serial). "
             "Use $SLURM_CPUS_PER_TASK on the cluster.",
    )
    args = ap.parse_args()

    if not args.root.exists():
        print(f"error: --root {args.root} does not exist", file=sys.stderr)
        return 2

    # Build the full task list up front so we can show progress.
    targets = list(_iter_targets(args.root))
    n_total = len(targets)
    if n_total == 0:
        print(f"No .npy files matching the compression specs found under {args.root}")
        return 0

    total_before = 0
    total_after = 0
    n_converted = 0
    n_skipped = 0
    n_errors = 0
    n_done = 0

    def _handle_result(res):
        nonlocal total_before, total_after, n_converted, n_skipped, n_errors, n_done
        n_done += 1
        npy_path, before, after, err = res
        if err is not None:
            n_errors += 1
            print(f"ERROR {npy_path}: {err}", file=sys.stderr)
            return
        total_before += before
        total_after += after
        if after == 0 and not args.dry_run:
            n_skipped += 1
        else:
            n_converted += 1
        if not args.quiet:
            if args.dry_run:
                print(f"[dry-run] would convert {npy_path} ({_format_bytes(before)})")
            else:
                ratio = (after / before) if before > 0 else 0.0
                print(
                    f"[{n_done}/{n_total}] "
                    f"{npy_path.relative_to(args.root)}: "
                    f"{_format_bytes(before)} -> {_format_bytes(after)} "
                    f"({ratio*100:.0f}%)",
                    flush=True,
                )

    tasks = [
        (p, dtype, not args.keep_npy, args.dry_run)
        for (p, dtype) in targets
    ]

    if args.workers <= 1:
        for t in tasks:
            _handle_result(_convert_worker(t))
    else:
        # Use imap_unordered so short files finish while long ones are still running.
        with mp.Pool(args.workers) as pool:
            for res in pool.imap_unordered(_convert_worker, tasks, chunksize=4):
                _handle_result(res)

    print()
    print("=" * 60)
    if args.dry_run:
        print(f"Dry run: {n_converted} files totalling {_format_bytes(total_before)}")
        print("Re-run without --dry-run to convert.")
    else:
        saved = total_before - total_after
        ratio = (total_after / total_before) if total_before > 0 else 0.0
        print(f"Converted: {n_converted} files  (workers={args.workers})")
        print(f"Skipped (already .npz): {n_skipped}")
        print(f"Errors: {n_errors}")
        print(f"Before: {_format_bytes(total_before)}")
        print(f"After:  {_format_bytes(total_after)}  ({ratio*100:.0f}% of original)")
        print(f"Freed:  {_format_bytes(saved)}")

    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
