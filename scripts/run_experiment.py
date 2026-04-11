"""Main experiment runner for the graphical horseshoe project.

Usage:
    python scripts/run_experiment.py --p 50 --T 250 --method nuts
    python scripts/run_experiment.py --p 50 --T 250 --method advi --guide mean_field
    python scripts/run_experiment.py --synthetic --p 50 --T 250 --sparsity 0.10
"""

import argparse
import json
import os
import time

import numpy as np

from src.benchmarks.frequentist import run_glasso, run_ledoit_wolf, run_sample_cov
from src.evaluation.metrics import frobenius_loss, spectral_loss, sparsity_metrics, steins_loss
from src.utils.matrix_utils import (
    sample_data_from_omega,
    sparse_omega_band,
    sparse_omega_block_diagonal,
    sparse_omega_erdos_renyi,
)


GRAPH_GENERATORS = {
    "erdos_renyi": sparse_omega_erdos_renyi,
    "band": sparse_omega_band,
    "block_diagonal": sparse_omega_block_diagonal,
}


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
                        choices=["erdos_renyi", "band", "block_diagonal"],
                        help="Graph structure for synthetic data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/",
                        help="Directory to save results")

    # NUTS-specific
    parser.add_argument("--num-warmup", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.85)

    # ADVI-specific
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-seeds", type=int, default=5)

    return parser.parse_args()


def generate_synthetic_data(args):
    gen_fn = GRAPH_GENERATORS[args.graph]
    kwargs = {"p": args.p, "seed": args.seed}
    if args.graph == "erdos_renyi":
        kwargs["sparsity"] = args.sparsity
    elif args.graph == "block_diagonal":
        kwargs["intra_sparsity"] = args.sparsity

    Omega_true, edge_set, _ = gen_fn(**kwargs)
    Y = sample_data_from_omega(Omega_true, T=args.T, seed=args.seed + 1)
    return Y, Omega_true, edge_set


def evaluate(Omega_hat, Omega_true, method_name):
    results = {
        "method": method_name,
        "steins_loss": steins_loss(Omega_hat, Omega_true),
        "frobenius_loss": frobenius_loss(Omega_hat, Omega_true),
        "spectral_loss": spectral_loss(Omega_hat, Omega_true),
    }
    results.update(sparsity_metrics(Omega_hat, Omega_true))
    return results


def run_bayesian_method(args, Y):
    import jax.numpy as jnp
    from src.models.graphical_horseshoe import graphical_horseshoe

    if args.method == "nuts":
        from src.inference.nuts_runner import extract_omega_samples, run_nuts

        mcmc = run_nuts(
            model=graphical_horseshoe,
            Y=Y,
            p=args.p,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            target_accept_prob=args.target_accept,
            rng_seed=args.seed,
        )
        Omega_samples = extract_omega_samples(mcmc, args.p)
        Omega_hat = np.array(jnp.mean(Omega_samples, axis=0))
        return Omega_hat, {"mcmc": mcmc, "Omega_samples": np.array(Omega_samples)}

    elif args.method == "advi":
        from src.inference.advi_runner import run_advi

        result = run_advi(
            model=graphical_horseshoe,
            Y=Y,
            p=args.p,
            guide_type=args.guide,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            num_seeds=args.num_seeds,
            rng_seed=args.seed,
        )
        samples = result["samples"]
        if "omega_offdiag" in samples:
            offdiag = np.array(samples["omega_offdiag"])
        else:
            offdiag = np.array(samples["z"] * samples["lambdas"]
                               * samples["tau"][:, None])
        diag = np.array(samples["omega_diag"])

        p = args.p
        idx_upper = np.triu_indices(p, k=1)
        Omega_mean = np.zeros((p, p))
        Omega_mean[idx_upper] = offdiag.mean(axis=0)
        Omega_mean = Omega_mean + Omega_mean.T + np.diag(diag.mean(axis=0))
        return Omega_mean, result


def run_frequentist_method(args, Y):
    if args.method == "glasso":
        Sigma, Omega, alpha = run_glasso(Y)
        return Omega, {"alpha_selected": alpha}
    elif args.method == "ledoit_wolf":
        Sigma, Omega, shrinkage = run_ledoit_wolf(Y)
        return Omega, {"shrinkage_intensity": shrinkage}
    elif args.method == "sample_cov":
        Sigma, Omega = run_sample_cov(Y)
        return Omega, {}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate or load data
    if args.synthetic:
        Y, Omega_true, edge_set = generate_synthetic_data(args)
        print(f"Synthetic data: p={args.p}, T={args.T}, graph={args.graph}, "
              f"sparsity={args.sparsity}, edges={len(edge_set)}")
    else:
        data_path = f"data/processed/returns_p{args.p}.npy"
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Processed return data not found at {data_path}. "
                "Run data preprocessing first or use --synthetic."
            )
        Y = np.load(data_path)[:args.T]
        Omega_true = None

    methods_to_run = (
        ["nuts", "advi", "glasso", "ledoit_wolf", "sample_cov"]
        if args.method == "all"
        else [args.method]
    )

    all_results = []
    for method in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {method}")
        print(f"{'='*60}")

        args.method = method
        start = time.time()

        if method in ("nuts", "advi"):
            Omega_hat, extra = run_bayesian_method(args, Y)
        else:
            Omega_hat, extra = run_frequentist_method(args, Y)

        elapsed = time.time() - start

        if Omega_hat is not None and Omega_true is not None:
            metrics = evaluate(Omega_hat, Omega_true, method)
            metrics["elapsed_seconds"] = elapsed
            all_results.append(metrics)
            print(f"  Stein's loss:    {metrics['steins_loss']:.4f}")
            print(f"  Frobenius loss:  {metrics['frobenius_loss']:.4f}")
            print(f"  Spectral loss:   {metrics['spectral_loss']:.4f}")
            print(f"  F1:              {metrics['f1']:.4f}")
            print(f"  MCC:             {metrics['mcc']:.4f}")
            print(f"  Time:            {elapsed:.1f}s")

        # Save Omega estimate
        tag = f"{method}_p{args.p}_T{args.T}_{args.graph}_s{args.seed}"
        if Omega_hat is not None:
            np.save(os.path.join(args.output_dir, f"omega_{tag}.npy"), Omega_hat)

    # Save metrics summary
    if all_results:
        summary_path = os.path.join(args.output_dir, f"metrics_p{args.p}_T{args.T}.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nMetrics saved to {summary_path}")


if __name__ == "__main__":
    main()
