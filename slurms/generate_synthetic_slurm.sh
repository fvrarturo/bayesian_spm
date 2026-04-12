#!/bin/bash
#SBATCH --job-name=gen_synth
#SBATCH --array=0-83
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err

# ======================================================================
# Synthetic data generation for the sparse Bayesian precision matrix
# estimation project (6.7830).
#
# Each array task processes ALL seeds for a single config_id in the
# manifest.  With 84 configs and 20 seeds each, this produces 1680
# (Omega, Sigma, Y, metadata) triples distributed across 84 tasks.
#
# Resource notes
# --------------
# Every task is single-core, needs <4GB of RAM, and runs in well under
# a minute.  We pad the wall time to 15 minutes to absorb cold-start
# overhead (conda activation, first-time numpy imports, filesystem
# metadata syncs) on the mit_normal partition.
#
# Run
# ---
#   mkdir -p logs
#   python scripts/generate_config_manifest.py   # one-time
#   sbatch scripts/generate_synthetic_slurm.sh
#
# To regenerate a subset instead of the full grid, submit a custom
# --array range, e.g. `sbatch --array=0,7,42 scripts/generate_synthetic_slurm.sh`.
# ======================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/synthetic

# --- Resolve a Python interpreter -------------------------------------------
# Prefer an already-activated conda env; otherwise try a few fallbacks so
# the script is robust to different cluster login environments.
PYTHON="${HOME}/.conda/envs/ggm_horseshoe/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    # Try to activate via module load + conda activate.
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
    # `conda activate` requires the shell hook.
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate ggm_horseshoe 2>/dev/null || true
    fi
    PYTHON=python
fi

echo "=== Host: $(hostname) ==="
echo "=== Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA} ==="
echo "=== Python: $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])') ==="

MANIFEST="data/synthetic/configs/config_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found at $MANIFEST -- generating it first."
    $PYTHON scripts/generate_config_manifest.py
fi

# --- Run ----------------------------------------------------------------------
$PYTHON scripts/generate_synthetic_data.py \
    --config-id "${SLURM_ARRAY_TASK_ID}" \
    --manifest "$MANIFEST" \
    --output-dir data/synthetic \
    --summary-path "logs/summary_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.json" \
    --log-level INFO

echo "=== Task ${SLURM_ARRAY_TASK_ID:-0} finished ==="
