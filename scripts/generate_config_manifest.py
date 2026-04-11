"""Write the experiment config manifest to disk.

This is a thin wrapper around ``src.utils.configs.compute_configs``.
Running this script produces the canonical
``data/synthetic/configs/config_manifest.json`` file, which every
downstream script (generation, inference, evaluation) uses to
discover what configurations exist.

Usage
-----
    python scripts/generate_config_manifest.py
    python scripts/generate_config_manifest.py --output path/to/manifest.json
    python scripts/generate_config_manifest.py --print  # also print to stdout
"""

import argparse
import json
import sys
from pathlib import Path

# Allow "python scripts/<name>.py" from the repo root to find src/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.configs import compute_configs, expected_config_count  # noqa: E402


DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "synthetic" / "configs" / "config_manifest.json"


def build_manifest(output_path: Path) -> list:
    """Compute configs and write them to ``output_path`` as JSON.

    Returns the list of configs that was written.
    """
    configs = compute_configs()

    expected = expected_config_count()
    if len(configs) != expected:
        raise RuntimeError(
            f"Config count {len(configs)} != expected {expected}. "
            "Grid logic is inconsistent."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(configs, f, indent=2)

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the config manifest for synthetic data generation."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Output path (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print a human-readable summary to stdout.",
    )
    args = parser.parse_args()

    configs = build_manifest(args.output)

    print(f"Wrote {len(configs)} configs to {args.output}")

    if args.print:
        print()
        print(f"{'id':>4}  {'p':>4}  {'gamma':>6}  {'T':>5}  "
              f"{'graph':<15}  {'s':>6}  n_blocks  n_seeds")
        print("-" * 72)
        for c in configs:
            nb = c["n_blocks"] if c["n_blocks"] is not None else "-"
            print(
                f"{c['config_id']:>4}  {c['p']:>4}  {c['gamma']:>6.2f}  "
                f"{c['T']:>5}  {c['graph']:<15}  {c['sparsity']:>6.2f}  "
                f"{str(nb):>8}  {c['n_seeds']:>7}"
            )


if __name__ == "__main__":
    main()
