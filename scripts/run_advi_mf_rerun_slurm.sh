#!/bin/bash
#SBATCH --job-name=advi_mf_rerun
#SBATCH --array=0-209%50
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/advi_mf_rerun_%A_%a.out
#SBATCH --error=logs/advi_mf_rerun_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PYTHON="${HOME}/.conda/envs/ggm_horseshoe/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load miniforge 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate ggm_horseshoe
    PYTHON=python
fi

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== task_id=${SLURM_ARRAY_TASK_ID} ==="

$PYTHON scripts/run_inference_single.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --task-manifest results/task_manifests/advi_mf_tier2.json \
    --log-level INFO

echo "=== Task ${SLURM_ARRAY_TASK_ID} finished ==="
