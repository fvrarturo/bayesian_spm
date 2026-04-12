#!/bin/bash
#SBATCH --job-name=nuts_infer
#SBATCH --array=0-1679%100
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nuts_%A_%a.out
#SBATCH --error=logs/nuts_%A_%a.err

# =====================================================================
# NUTS inference on the WORK1 synthetic data.
#
# Each task processes ONE (config, seed) pair.  NUTS is the slowest
# and most memory-intensive method in the pipeline; wall-time budget is
# 4 hours and memory is 16 GB.  Many p=100 tasks will hit the wall and
# report status="timeout" — this is an expected finding, not a bug.
#
# Full-grid size: 84 configs × 20 seeds = 1,680 tasks.
# The %100 caps concurrency to 100 simultaneous tasks.
#
# For the progress-report slice:
#     python scripts/generate_task_manifests.py --tier 1
#     sbatch --array=0-9 scripts/run_nuts_slurm.sh    # 2 cfgs × 5 seeds
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

# Tell JAX to use a single CPU thread — NumPyro's multi-chain MCMC
# parallelises differently and extra threads only hurt.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Force unbuffered stdout/stderr so any print() shows up in .out immediately.
export PYTHONUNBUFFERED=1

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA} ==="
echo "=== Python: $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])') ==="

MANIFEST="results/task_manifests/nuts.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "NUTS task manifest not found at $MANIFEST. Regenerating tier 3."
    $PYTHON scripts/generate_task_manifests.py --tier 3
fi

$PYTHON scripts/run_inference_single.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --task-manifest "$MANIFEST" \
    --data-root data/synthetic \
    --results-root results/synthetic \
    --config-manifest data/synthetic/configs/config_manifest.json \
    --timeout-seconds 14400 \
    --log-level INFO

echo "=== Task ${SLURM_ARRAY_TASK_ID:-0} finished ==="
