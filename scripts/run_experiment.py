"""Main experiment runner for the graphical horseshoe project.

Usage:
    python scripts/run_experiment.py --p 50 --T 250 --method nuts
    python scripts/run_experiment.py --p 50 --T 250 --method advi --guide mean_field
    python scripts/run_experiment.py --synthetic --p 50 --T 250 --sparsity 0.10
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run graphical horseshoe experiments")
    parser.add_argument("--p", type=int, default=50, help="Dimension")
    parser.add_argument("--T", type=int, default=250, help="Sample size")
    parser.add_argument("--method", type=str, default="nuts",
                        choices=["nuts", "advi", "glasso", "ledoit_wolf", "sample_cov", "all"],
                        help="Inference method")
    parser.add_argument("--guide", type=str, default="mean_field",
                        choices=["mean_field", "full_rank", "low_rank", "map"],
                        help="ADVI guide type")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of CRSP returns")
    parser.add_argument("--sparsity", type=float, default=0.10,
                        help="Sparsity level for synthetic data")
    parser.add_argument("--graph", type=str, default="erdos_renyi",
                        choices=["erdos_renyi", "band", "block"],
                        help="Graph structure for synthetic data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/",
                        help="Directory to save results")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("Experiment runner not yet implemented")


if __name__ == "__main__":
    main()
