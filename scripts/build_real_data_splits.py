"""Build train/test splits for the FF48 real-data experiment (WORK4 §3.2).

Reads ``data/real/ff48/Y.npy`` (produced by ``preprocess_real_data.py``)
and slices it into one or more rolling-window (train, test) pairs.  Each
window is materialised as a self-contained directory that the existing
inference + evaluation pipeline can consume without modification:

    data/real/ff48/window_<NN>/
        Y.npy           (T_train, 48)   training data (the runner reads this)
        Y_test.npy      (T_test, 48)    held-out data for OOS metrics
        omega_true.npy  (48, 48) = I    placeholder; real_data sentinel below
        metadata.json   real_data=True, train/test dates, dir_path, etc.

The window directories slot in as "configs" in a parallel manifest:

    data/real/ff48/configs/config_manifest_real.json   one entry per window
    results/task_manifests/real_data.json              one task per window×method

The runner key invariant is that ``data_dir/Y.npy`` is what gets fit; the
test data is read by the (real-data-aware) evaluator from ``Y_test.npy``.

Modes
-----
    --mode single        : one window, train on first ``train-size`` rows,
                           test on next ``test-size`` rows.
    --mode rolling       : ``n-windows`` non-overlapping windows, each with
                           ``train-size`` train + ``test-size`` test rows,
                           advancing by ``test-size`` (i.e. test slices tile
                           the timeline without gaps).

Usage
-----
    # Single split (default), most recent ~2 years
    python scripts/build_real_data_splits.py
    # Rolling windows, 12 of them, ~3 years of test coverage
    python scripts/build_real_data_splits.py --mode rolling --n-windows 12
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "real" / "ff48"
DEFAULT_CONFIG_DIR = REPO_ROOT / "data" / "real" / "ff48" / "configs"
DEFAULT_TASK_MANIFEST = REPO_ROOT / "results" / "task_manifests" / "real_data.json"
DEFAULT_METHODS = ("gibbs", "advi_mf", "glasso", "nuts")


def _slice_dates(dates: List[str], idx: np.ndarray) -> List[str]:
    return [dates[i] for i in idx]


def _build_windows(
    n_rows: int,
    mode: str,
    train_size: int,
    test_size: int,
    n_windows: int,
    align_end: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute (train_idx, test_idx) integer-index pairs.

    For ``rolling``, windows are anchored at the END of the data (most
    recent test windows are the LAST ``test_size`` rows, and earlier
    windows step backward by ``test_size``).  This keeps the most recent
    market regime in the test set.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []
    if mode == "single":
        if align_end:
            test_end = n_rows
            test_start = test_end - test_size
        else:
            test_start = train_size
            test_end = train_size + test_size
        train_start = test_start - train_size
        if train_start < 0:
            raise ValueError(
                f"not enough data: need train_size+test_size={train_size + test_size} "
                f"rows, have {n_rows}"
            )
        out.append(
            (np.arange(train_start, test_start), np.arange(test_start, test_end))
        )
        return out

    if mode == "rolling":
        # Anchor the LAST window's test set at the end of the data, then
        # step backwards by test_size for each earlier window.
        test_end = n_rows
        for w in range(n_windows):
            t_end = test_end - w * test_size
            t_start = t_end - test_size
            tr_start = t_start - train_size
            if tr_start < 0:
                # Truncate: we ran out of history.
                break
            out.append(
                (np.arange(tr_start, t_start), np.arange(t_start, t_end))
            )
        # Order chronologically (oldest first) so window indices grow with time.
        out.reverse()
        return out

    raise ValueError(f"unknown mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Directory containing the FF48 Y.npy + metadata.json from the preprocessor.",
    )
    ap.add_argument(
        "--config-dir", type=Path, default=DEFAULT_CONFIG_DIR,
        help="Where to write config_manifest_real.json.",
    )
    ap.add_argument(
        "--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST,
        help="Where to write the SLURM task manifest (one task per window × method).",
    )
    ap.add_argument(
        "--mode", choices=["single", "rolling"], default="single",
        help="Split scheme.  'single' = one (train, test) pair; "
             "'rolling' = n-windows non-overlapping pairs.",
    )
    ap.add_argument(
        "--train-size", type=int, default=250,
        help="Trading days in each training window.  Default 250 (≈ 1 year).",
    )
    ap.add_argument(
        "--test-size", type=int, default=250,
        help="Trading days in each test window.  Default 250 (≈ 1 year).",
    )
    ap.add_argument(
        "--n-windows", type=int, default=12,
        help="Number of rolling windows (only used with --mode rolling).",
    )
    ap.add_argument(
        "--align-end", action="store_true", default=True,
        help="Anchor the test window at the end of the data (most-recent regime). "
             "Default: True.",
    )
    ap.add_argument(
        "--methods", type=str, default=",".join(DEFAULT_METHODS),
        help="Comma-separated list of methods to enumerate in the task manifest.",
    )
    args = ap.parse_args()

    # 1. Load Y + metadata
    Y = np.load(args.data_dir / "Y.npy")
    with open(args.data_dir / "metadata.json") as f:
        ff48_meta = json.load(f)
    p = int(Y.shape[1])
    n_rows = int(Y.shape[0])
    if p != 48:
        print(f"[splits] warning: expected p=48, got p={p}")
    print(f"[splits] loaded Y: {Y.shape}, dates "
          f"{ff48_meta.get('first_date')} – {ff48_meta.get('last_date')}")

    # 2. Build windows
    windows = _build_windows(
        n_rows=n_rows,
        mode=args.mode,
        train_size=args.train_size,
        test_size=args.test_size,
        n_windows=args.n_windows,
        align_end=args.align_end,
    )
    if not windows:
        raise RuntimeError("no windows produced")
    print(f"[splits] mode={args.mode}, built {len(windows)} window(s)")

    # 3. Materialise each window directory
    industries = ff48_meta.get("industries", [])
    # Need to recover the date strings.  The preprocessor records first/last
    # date but not the per-row dates.  For this script, we don't strictly
    # need them — just record the train/test row-index ranges and the
    # corresponding first/last dates if recoverable.
    configs = []
    for w, (tr_idx, te_idx) in enumerate(windows):
        wdir = args.data_dir / f"window_{w:02d}"
        wdir.mkdir(parents=True, exist_ok=True)
        Y_train = Y[tr_idx]
        Y_test = Y[te_idx]
        np.save(wdir / "Y.npy", Y_train)
        np.save(wdir / "Y_test.npy", Y_test)
        # Placeholder ground truth so the existing data-loading code is happy.
        np.save(wdir / "omega_true.npy", np.eye(p, dtype=np.float64))

        T_train = int(Y_train.shape[0])
        T_test = int(Y_test.shape[0])
        gamma = float(p / T_train)
        wmeta = {
            "real_data": True,
            "source": ff48_meta.get("source", "FF48"),
            # ``config_id`` and ``seed`` are required by run_single.py's data
            # loader (it reads them from metadata for the diagnostics record).
            # For real data, config_id == window_id and seed == 0.
            "config_id": w,
            "seed": 0,
            "data_seed": 0,
            "window_id": w,
            "p": p,
            "T": T_train,
            "T_test": T_test,
            "gamma": gamma,
            "graph": "ff48",
            "sparsity": None,
            "train_idx_start": int(tr_idx[0]),
            "train_idx_end": int(tr_idx[-1]) + 1,
            "test_idx_start": int(te_idx[0]),
            "test_idx_end": int(te_idx[-1]) + 1,
            "industries": industries,
            "split_mode": args.mode,
        }
        with open(wdir / "metadata.json", "w") as f:
            json.dump(wmeta, f, indent=2)

        # Also record one entry in the parallel config manifest.
        configs.append({
            "config_id": w,
            "p": p,
            "T": T_train,
            "gamma": gamma,
            "graph": "ff48",
            "sparsity": None,
            "n_seeds": 1,
            # ``dir_path`` is *relative to data-root*; the existing infra
            # composes ``data_root / dir_path / dir_name_seed(seed)`` to
            # locate the seed dir.  For real data we want the WINDOW dir
            # itself to be the seed dir, so we set dir_path to the window
            # path and put the data files at the seed-0 sublevel.
            "dir_path": f"real/ff48/window_{w:02d}",
        })

    # 3b. The runner expects a `seed_<NN>` subdirectory inside ``dir_path``;
    # but for real data we don't have multiple seeds.  Easiest: also
    # write the same files into ``window_NN/seed_00/`` as a symlink-free
    # "single seed" so the standard data-loading path works unchanged.
    for w, _ in enumerate(windows):
        wdir = args.data_dir / f"window_{w:02d}"
        sdir = wdir / "seed_00"
        sdir.mkdir(parents=True, exist_ok=True)
        for fn in ("Y.npy", "Y_test.npy", "omega_true.npy", "metadata.json"):
            src = wdir / fn
            dst = sdir / fn
            if dst.exists():
                continue
            # Create a hardlink (no extra disk usage).
            try:
                dst.hardlink_to(src)
            except (AttributeError, OSError):
                # Python < 3.10 or filesystem doesn't support hardlinks: copy
                import shutil
                shutil.copyfile(src, dst)

    # 4. Save config manifest
    args.config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.config_dir / "config_manifest_real.json"
    with open(cfg_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"[splits] wrote {len(configs)} configs -> {cfg_path}")

    # 5. Save task manifest (one task per window × method)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tasks = []
    tid = 0
    for c in configs:
        for m in methods:
            tasks.append({
                "task_id": tid,
                "config_id": int(c["config_id"]),
                "seed": 0,
                "method": m,
            })
            tid += 1
    args.task_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.task_manifest, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"[splits] wrote {len(tasks)} tasks -> {args.task_manifest}")

    # 6. Summary
    print(f"\n[splits] summary:")
    for c in configs:
        print(f"  window {c['config_id']:02d}: dir={c['dir_path']}  "
              f"T_train={c['T']}  γ={c['gamma']:.3f}")
    print(f"\n[splits] tasks per method × {len(configs)} windows = {len(tasks)} total tasks")
    print(f"[splits] Methods: {methods}")
    print(f"\n[splits] DONE.  Submit with:")
    print(f"  sbatch --array=0-{len(tasks)-1} scripts/run_real_data_slurm.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
