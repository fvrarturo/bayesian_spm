"""Experimental grid for the synthetic data generation pipeline.

This module is the single source of truth for WHICH configurations exist.
The ``compute_configs`` function is pure: it takes no arguments and
returns a deterministic list of config dictionaries.  Both the manifest-
building script and the test suite import it.

See ``_info/WORK1.md`` for the full design rationale.
"""

import itertools
import math
from typing import List, Optional

# ----------------------------------------------------------------------
# Experimental grid (see WORK1.md §2)
# ----------------------------------------------------------------------

P_VALUES = [10, 50, 100]
GAMMA_VALUES = [0.90, 0.67, 0.42, 0.20, 0.10]
GRAPHS = ["erdos_renyi", "block_diagonal"]
SPARSITY_VALUES = [0.05, 0.10, 0.30]
SIGNAL_RANGE = [0.3, 0.8]
N_SEEDS = 20

# Number of blocks for the block-diagonal graph generator, keyed by p.
N_BLOCKS_MAP = {10: 2, 50: 5, 100: 5}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def T_from_gamma(p: int, gamma: float) -> int:
    """Compute the sample size T = ceil(p / gamma)."""
    return int(math.ceil(p / gamma))


def should_skip(sparsity: float, gamma: float) -> bool:
    """Return True iff this (sparsity, gamma) pair is in the skip list.

    We skip the (s=0.30, gamma=0.90) combination for every graph and p
    because the diagonal shift needed to enforce positive definiteness
    at such high edge density becomes pathologically large, and the
    estimation problem is uninformative at such a high concentration
    ratio. See WORK1.md §2.3.
    """
    return sparsity == 0.30 and gamma == 0.90


def dir_name_gamma(gamma: float) -> str:
    """Encode a gamma value as a directory-safe string like 'gamma090'."""
    return f"gamma{int(round(gamma * 100)):03d}"


def dir_name_sparsity(sparsity: float) -> str:
    """Encode a sparsity value as a directory-safe string like 's010'."""
    return f"s{int(round(sparsity * 100)):03d}"


def dir_name_p(p: int) -> str:
    """Encode a dimension as a zero-padded directory name like 'p050'."""
    return f"p{p:03d}"


def dir_name_seed(seed: int) -> str:
    """Encode a seed as a zero-padded directory name like 'seed_03'."""
    return f"seed_{seed:02d}"


def dir_path_for_config(graph: str, p: int, gamma: float, sparsity: float) -> str:
    """Return the relative directory path for a config (no seed)."""
    return "/".join(
        [
            graph,
            dir_name_p(p),
            dir_name_gamma(gamma),
            dir_name_sparsity(sparsity),
        ]
    )


# ----------------------------------------------------------------------
# Core: build the list of configurations
# ----------------------------------------------------------------------

def compute_configs() -> List[dict]:
    """Return the full list of valid experiment configurations.

    Order: (p, gamma, graph, sparsity) in the Cartesian product, with
    skipped combinations removed.  Each config is assigned a unique
    integer ``config_id`` starting from 0.  With the default grid this
    yields exactly 84 configs.

    Returns
    -------
    List[dict]
        Each dict has keys:
        ``config_id, p, gamma, T, graph, sparsity, n_blocks,
        signal_range, n_seeds, skip, dir_path``.
    """
    configs: List[dict] = []
    config_id = 0

    for p, gamma, graph, sparsity in itertools.product(
        P_VALUES, GAMMA_VALUES, GRAPHS, SPARSITY_VALUES
    ):
        if should_skip(sparsity, gamma):
            continue

        T = T_from_gamma(p, gamma)
        n_blocks: Optional[int] = (
            N_BLOCKS_MAP[p] if graph == "block_diagonal" else None
        )
        dir_path = dir_path_for_config(graph, p, gamma, sparsity)

        configs.append(
            {
                "config_id": config_id,
                "p": p,
                "gamma": gamma,
                "T": T,
                "graph": graph,
                "sparsity": sparsity,
                "n_blocks": n_blocks,
                "signal_range": list(SIGNAL_RANGE),
                "n_seeds": N_SEEDS,
                "skip": False,
                "dir_path": dir_path,
            }
        )
        config_id += 1

    return configs


def expected_config_count() -> int:
    """Return the expected number of configs after skips are applied.

    Useful for tests that want to assert the grid hasn't silently
    changed size.
    """
    total = len(P_VALUES) * len(GAMMA_VALUES) * len(GRAPHS) * len(SPARSITY_VALUES)
    n_skipped = sum(
        1
        for s, g in itertools.product(SPARSITY_VALUES, GAMMA_VALUES)
        if should_skip(s, g)
    )
    # One skipped (s, gamma) pair removes one config per (p, graph) combo.
    per_skip = len(P_VALUES) * len(GRAPHS)
    return total - n_skipped * per_skip
