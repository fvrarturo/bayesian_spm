#!/bin/bash
#SBATCH --job-name=advi_infer
#SBATCH --array=0-3359%50
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/advi_%A_%a.out
#SBATCH --error=logs/advi_%A_%a.err

# =====================================================================
# ADVI inference on the WORK1 synthetic data.
#
# Each task processes ONE (config, seed, method) triple, where method
# is advi_mf or advi_fr.  Full-rank ADVI auto-falls-back to low-rank at
# p=100 (memory guard inside run_single.py).
#
# Full-grid size: 84 configs × 20 seeds × 2 methods = 3,360 tasks.
# The %50 caps concurrency to 50 simultaneous tasks to stay polite on
# the shared cluster.  Remove the %50 for uncapped submission.
#
# Before submitting: regenerate the task manifest for the tier you want.
#     python scripts/generate_task_manifests.py --tier 3          # full
#     sbatch scripts/run_advi_slurm.sh
#
#     python scripts/generate_task_manifests.py --tier 1          # progress report
#     sbatch --array=0-19 scripts/run_advi_slurm.sh               # 2 cfgs × 5 seeds × 2 methods
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

# JAX threading: ADVI is sequential per sample, extra CPU threads give no speedup.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Force unbuffered stdout/stderr so per-step progress prints appear in the
# .out file while the job is running (not only at exit).
export PYTHONUNBUFFERED=1

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA} ==="
echo "=== Python: $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])') ==="

MANIFEST="results/task_manifests/advi.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ADVI task manifest not found at $MANIFEST. Regenerating tier 3."
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
