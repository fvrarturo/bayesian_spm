#!/bin/bash
#SBATCH --job-name=gibbs_infer
#SBATCH --array=0-1679%100
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gibbs_%A_%a.out
#SBATCH --error=logs/gibbs_%A_%a.err

# =====================================================================
# Gibbs sampler (Li et al. 2019) on the WORK1 synthetic data.
#
# Each task processes ONE (config, seed) pair.  The Gibbs sampler is
# pure-NumPy (no JAX), O(p^3) per sweep.
#
# Expected runtime:
#   p=10:   < 1 min     (fast batch)
#   p=50:   10–30 min
#   p=100:  1–3 hours   (may succeed where NUTS times out)
#
# Resource allocation: 1 CPU, 8 GB RAM, 4-hour wall.
# =====================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/synthetic results/task_manifests

PYTHON="${HOME}/.conda/envs/ggm_horseshoe/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate ggm_horseshoe 2>/dev/null || true
    fi
    PYTHON=python
fi

export PYTHONUNBUFFERED=1
# No JAX/XLA settings needed — Gibbs is pure NumPy.

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA} ==="
echo "=== Python: $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])') ==="

MANIFEST="results/task_manifests/gibbs.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Gibbs task manifest not found at $MANIFEST. Regenerating tier 3."
    $PYTHON scripts/generate_task_manifests.py --tier 3
fi

$PYTHON scripts/run_inference_single.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --task-manifest "$MANIFEST" \
    --data-root data/synthetic \
    --results-root results/synthetic \
    --config-manifest data/synthetic/configs/config_manifest.json \
    --log-level INFO

echo "=== Task ${SLURM_ARRAY_TASK_ID:-0} finished ==="
