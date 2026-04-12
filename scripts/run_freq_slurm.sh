#!/bin/bash
#SBATCH --job-name=freq_infer
#SBATCH --array=0-83
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/freq_%A_%a.out
#SBATCH --error=logs/freq_%A_%a.err

# =====================================================================
# Frequentist inference on the WORK1 synthetic data.
#
# Each array task processes ONE config, running all three frequentist
# methods (sample_cov, ledoit_wolf, glasso) across all 20 seeds.  This
# is the "fast lane": every method takes well under a minute.
#
# Override --array to run a subset:
#     sbatch --array=0-3 scripts/run_freq_slurm.sh   # first 4 configs
#     sbatch --array=0,7,42 scripts/run_freq_slurm.sh  # specific ids
#
# For the progress-report tier (2 configs), run:
#     python scripts/generate_task_manifests.py --tier 1
#     sbatch --array=0-1 scripts/run_freq_slurm.sh
# =====================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/synthetic results/task_manifests

# Python resolution fallback (same pattern as generate_synthetic_slurm.sh).
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

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA} ==="
echo "=== Python: $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])') ==="

$PYTHON scripts/run_inference_single.py \
    --config-id "${SLURM_ARRAY_TASK_ID}" \
    --methods sample_cov,ledoit_wolf,glasso \
    --seeds all \
    --data-root data/synthetic \
    --results-root results/synthetic \
    --config-manifest data/synthetic/configs/config_manifest.json \
    --log-level INFO

echo "=== Task ${SLURM_ARRAY_TASK_ID:-0} finished ==="
