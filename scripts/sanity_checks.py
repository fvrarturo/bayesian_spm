"""Optional post-generation sanity checks for the synthetic data tree.

These checks supplement ``audit_synthetic_data.py``: the audit verifies
*structural* integrity (files exist, shapes are right, metadata matches
the manifest), while this script characterises the *statistical*
properties of the generated dataset (edge-count distributions,
condition numbers, cross-seed independence, etc.).

Each check is implemented as a function returning ``None`` and printing
to stdout.  Run them individually or all at once.

Usage
-----
    # Run every check (default)
    python scripts/sanity_checks.py

    # Run a single check by number
    python scripts/sanity_checks.py --check 3

    # Run a subset
    python scripts/sanity_checks.py --check 1 --check 4 --check 8

    # Use a non-default data root
    python scripts/sanity_checks.py --data-root /custom/path

The checks
----------
    1. edge_counts          — mean/std/min/max edges per (graph, p, s)
    2. condition_numbers    — median/max condition number per (graph, p, s)
    3. eigenvalue_floor     — verify all min eigenvalues >= 0.1 (must be 0)
    4. warnings_aggregate   — count of warnings across all 1680 seeds
    5. cross_seed_indep     — pairwise edge-set overlap for one config
    6. reconstruction       — Omega @ Sigma == I for 50 random seeds
    7. zero_mean_zscore     — distribution of per-seed sample-mean z-scores
    8. T_distribution       — verify all (p, T) pairs match WORK1 §2.2
"""

import argparse
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "synthetic"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _load_all_metadata(data_root: Path) -> List[dict]:
    """Walk the synthetic tree and load every metadata.json."""
    out: List[dict] = []
    for md_path in sorted(data_root.rglob("seed_*/metadata.json")):
        with open(md_path) as f:
            md = json.load(f)
        md["_path"] = str(md_path.parent)
        out.append(md)
    return out


def _hr(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ----------------------------------------------------------------------
# Check 1: aggregate edge-count distribution
# ----------------------------------------------------------------------

def check_edge_counts(data_root: Path) -> None:
    _hr("CHECK 1: edge-count distribution per (graph, p, s)")
    mds = _load_all_metadata(data_root)

    buckets: dict = defaultdict(list)
    for md in mds:
        key = (md["graph"], md["p"], md["sparsity"])
        buckets[key].append(md["n_edges"])

    print(
        f"{'graph':<16}{'p':>5}{'s':>7}{'n':>6}"
        f"{'mean':>10}{'std':>10}{'min':>7}{'max':>7}{'expected':>10}"
    )
    print("-" * 78)

    for key in sorted(buckets):
        graph, p, s = key
        es = buckets[key]
        if graph == "erdos_renyi":
            n_possible = p * (p - 1) // 2
            expected = s * n_possible
        else:  # block_diagonal
            from src.utils.configs import N_BLOCKS_MAP
            n_blocks = N_BLOCKS_MAP[p]
            block_size = p // n_blocks
            # Last block absorbs remainder, but for the p in {10,50,100}
            # the divisions are exact.
            n_within_block_pairs = n_blocks * block_size * (block_size - 1) // 2
            expected = s * n_within_block_pairs

        mean = st.mean(es)
        std = st.stdev(es) if len(es) > 1 else 0.0
        print(
            f"{graph:<16}{p:>5}{s:>7.2f}{len(es):>6}"
            f"{mean:>10.1f}{std:>10.1f}{min(es):>7}{max(es):>7}"
            f"{expected:>10.1f}"
        )


# ----------------------------------------------------------------------
# Check 2: condition-number distribution
# ----------------------------------------------------------------------

def check_condition_numbers(data_root: Path) -> None:
    _hr("CHECK 2: condition-number distribution per (graph, p, s)")
    mds = _load_all_metadata(data_root)

    buckets: dict = defaultdict(list)
    for md in mds:
        key = (md["graph"], md["p"], md["sparsity"])
        buckets[key].append(md["condition_number"])

    print(f"{'graph':<16}{'p':>5}{'s':>7}{'median':>12}{'max':>12}{'p95':>12}")
    print("-" * 64)
    for key in sorted(buckets):
        g, p, s = key
        ks = sorted(buckets[key])
        med = st.median(ks)
        mx = max(ks)
        p95 = ks[int(0.95 * len(ks))] if ks else 0
        print(f"{g:<16}{p:>5}{s:>7.2f}{med:>12.1f}{mx:>12.1f}{p95:>12.1f}")

    print()
    print(f"  Global max condition number: {max(md['condition_number'] for md in mds):.1f}")
    print("  (anything > 1000 should already have triggered a warning)")


# ----------------------------------------------------------------------
# Check 3: minimum eigenvalue floor
# ----------------------------------------------------------------------

def check_eigenvalue_floor(data_root: Path) -> None:
    _hr("CHECK 3: minimum eigenvalue >= 0.1 across all seeds")
    mds = _load_all_metadata(data_root)
    floor = 0.1
    tol = 1e-9

    bad = [
        (md["config_id"], md["seed"], md["min_eigenvalue"], md["_path"])
        for md in mds
        if md["min_eigenvalue"] < floor - tol
    ]

    print(f"  Total seeds checked:   {len(mds)}")
    print(f"  Below floor (<{floor}): {len(bad)}  (must be 0)")
    if bad:
        print("\n  First 10 violations:")
        for cid, seed, eig, path in bad[:10]:
            print(f"    cfg={cid:>3} seed={seed:>2}  eigmin={eig:.6e}  {path}")
    else:
        eigmins = [md["min_eigenvalue"] for md in mds]
        print(
            f"  Min eigenvalue across all seeds: "
            f"{min(eigmins):.10f}  (should be ~{floor})"
        )


# ----------------------------------------------------------------------
# Check 4: aggregate warnings
# ----------------------------------------------------------------------

def check_warnings_aggregate(data_root: Path) -> None:
    _hr("CHECK 4: aggregate warnings across all seeds")
    mds = _load_all_metadata(data_root)

    counts: Counter = Counter()
    seeds_with_warnings = 0
    per_config: dict = defaultdict(int)

    for md in mds:
        if md.get("warnings"):
            seeds_with_warnings += 1
            per_config[md["config_id"]] += len(md["warnings"])
            for w in md["warnings"]:
                counts[w.split(":")[0]] += 1

    print(f"  Total seeds:           {len(mds)}")
    print(f"  Seeds with warnings:   {seeds_with_warnings}")
    print(f"  Total warning lines:   {sum(counts.values())}")
    print()
    print("  Warning categories:")
    for kind, n in counts.most_common():
        print(f"    {n:>5}  {kind}")
    print()
    print("  Top 10 configs by warning count:")
    for cid, n in sorted(per_config.items(), key=lambda x: -x[1])[:10]:
        print(f"    cfg={cid:>3}  {n} warnings across 20 seeds")


# ----------------------------------------------------------------------
# Check 5: cross-seed independence (pairwise edge-set overlap)
# ----------------------------------------------------------------------

def check_cross_seed_independence(data_root: Path) -> None:
    _hr("CHECK 5: cross-seed independence (pairwise edge-set overlap)")
    # Pick a config with enough edges to make the comparison meaningful.
    target = data_root / "erdos_renyi" / "p050" / "gamma042" / "s030"
    if not target.exists():
        print(f"  Target config dir not found: {target}")
        print("  (skipping)")
        return

    seeds = sorted(target.glob("seed_*"))
    edge_sets = []
    for sd in seeds:
        with open(sd / "metadata.json") as f:
            md = json.load(f)
        edge_sets.append(frozenset(tuple(e) for e in md["edge_set"]))

    distinct = len(set(edge_sets))
    overlaps = []
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            overlaps.append(len(edge_sets[i] & edge_sets[j]))

    p, s = 50, 0.30
    n_possible = p * (p - 1) // 2
    expected_overlap = s * s * n_possible

    print(f"  Target config: erdos_renyi p=50 s=0.30 (config 40)")
    print(f"  Seeds inspected:           {len(edge_sets)}")
    print(f"  Distinct edge sets:        {distinct} (must equal n_seeds)")
    print()
    print(f"  Mean pairwise overlap:     {st.mean(overlaps):.1f}")
    print(f"  Median pairwise overlap:   {st.median(overlaps):.0f}")
    print(f"  Min pairwise overlap:      {min(overlaps)}")
    print(f"  Max pairwise overlap:      {max(overlaps)}")
    print(f"  Expected (s^2 * C(p,2)):   {expected_overlap:.1f}")


# ----------------------------------------------------------------------
# Check 6: post-hoc reconstruction (Omega @ Sigma == I)
# ----------------------------------------------------------------------

def check_reconstruction(data_root: Path, n_samples: int = 50) -> None:
    _hr(f"CHECK 6: Omega @ Sigma == I on {n_samples} random seeds")
    all_seeds = sorted(data_root.rglob("seed_*"))
    if len(all_seeds) == 0:
        print("  No seed dirs found.  (skipping)")
        return

    rng = random.Random(0)
    sample = rng.sample(all_seeds, min(n_samples, len(all_seeds)))
    worst = 0.0
    for sd in sample:
        Omega = np.load(sd / "omega_true.npy")
        Sigma = np.load(sd / "sigma_true.npy")
        err = float(np.max(np.abs(Omega @ Sigma - np.eye(Omega.shape[0]))))
        worst = max(worst, err)

    print(f"  Sampled seeds:                       {len(sample)}")
    print(f"  Worst max(|Omega @ Sigma - I|):      {worst:.2e}")
    print(f"  Tolerance (validation enforces):     1.00e-08")
    if worst > 1e-8:
        print("  STATUS: FAIL")
    else:
        print("  STATUS: ok")


# ----------------------------------------------------------------------
# Check 7: distribution of zero-mean z-scores
# ----------------------------------------------------------------------

def check_zero_mean_zscore(data_root: Path) -> None:
    _hr("CHECK 7: distribution of per-seed sample-mean z-scores")
    mds = _load_all_metadata(data_root)
    zs = [
        md["max_mean_zscore"]
        for md in mds
        if md.get("max_mean_zscore") is not None
        and not math.isnan(md["max_mean_zscore"])
    ]

    if not zs:
        print("  No z-scores found in metadata. (skipping)")
        return

    zs_sorted = sorted(zs)
    print(f"  Seeds with z-score:    {len(zs)}")
    print(f"  Mean max-z:            {st.mean(zs):.2f}")
    print(f"  Median max-z:          {st.median(zs):.2f}")
    print(f"  P90 max-z:             {zs_sorted[int(0.90 * len(zs))]:.2f}")
    print(f"  P99 max-z:             {zs_sorted[int(0.99 * len(zs))]:.2f}")
    print(f"  Max max-z:             {max(zs):.2f}")
    print(f"  Number > 5 (warning):  {sum(1 for z in zs if z > 5)}")
    print(f"  Number > 6:            {sum(1 for z in zs if z > 6)}")
    print(f"  Number > 7:            {sum(1 for z in zs if z > 7)}")


# ----------------------------------------------------------------------
# Check 8: distribution of (p, T) pairs
# ----------------------------------------------------------------------

def check_T_distribution(data_root: Path) -> None:
    _hr("CHECK 8: (p, T) distribution and seed counts")
    mds = _load_all_metadata(data_root)

    counts: Counter = Counter()
    gammas: dict = defaultdict(set)
    for md in mds:
        counts[(md["p"], md["T"])] += 1
        gammas[(md["p"], md["T"])].add(md["gamma"])

    print(f"{'(p, T)':<14}{'n_seeds':>10}{'gammas':>20}")
    print("-" * 44)
    for key in sorted(counts):
        p, T = key
        n = counts[key]
        gs = sorted(gammas[key])
        print(f"  ({p:>3}, {T:>4})  {n:>8}     {gs}")

    print()
    print(f"  Total (p, T) pairs: {len(counts)}  (expected 15)")
    print(f"  Total seeds:        {sum(counts.values())}  (expected 1680)")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

CHECKS = [
    ("edge_counts", check_edge_counts),
    ("condition_numbers", check_condition_numbers),
    ("eigenvalue_floor", check_eigenvalue_floor),
    ("warnings_aggregate", check_warnings_aggregate),
    ("cross_seed_indep", check_cross_seed_independence),
    ("reconstruction", check_reconstruction),
    ("zero_mean_zscore", check_zero_mean_zscore),
    ("T_distribution", check_T_distribution),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optional post-generation sanity checks for the synthetic data tree."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Synthetic data root directory (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--check",
        type=int,
        action="append",
        choices=list(range(1, len(CHECKS) + 1)),
        help="Run only this check (1..%d). Repeat to run several." % len(CHECKS),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available checks and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available checks:")
        for i, (name, _) in enumerate(CHECKS, 1):
            print(f"  {i}. {name}")
        return 0

    if not args.data_root.exists():
        print(f"ERROR: data root does not exist: {args.data_root}")
        return 1

    # Make src/ importable for check 1
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    selected = args.check if args.check else list(range(1, len(CHECKS) + 1))
    for idx in selected:
        name, fn = CHECKS[idx - 1]
        fn(args.data_root)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
