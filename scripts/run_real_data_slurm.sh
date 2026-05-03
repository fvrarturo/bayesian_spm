#!/bin/bash
#SBATCH --job-name=real_data_infer
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/real_data_%A_%a.out
#SBATCH --error=logs/real_data_%A_%a.err

# =====================================================================
# Real-data inference on the FF48 industry portfolios (WORK4 §3.5).
#
# Each task runs ONE (window, method) pair from
# ``results/task_manifests/real_data.json``.  The manifest is produced by
# ``scripts/build_real_data_splits.py``.
#
# Wall-time budget (per-task at p=48):
#   - glasso   ~1   s
#   - advi_mf  ~1   min
#   - gibbs    ~3-5 min
#   - nuts     ~30-60 min
#
# Submit:
#   sbatch --array=0-3 scripts/run_real_data_slurm.sh        # single window
#   sbatch --array=0-47 scripts/run_real_data_slurm.sh       # 12 rolling × 4 methods
# =====================================================================

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

# JAX/NumPy single-threaded so we don't fight SLURM's allocation.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== task_id=${SLURM_ARRAY_TASK_ID} ==="

# Real-data layout differs from the synthetic one: data lives at
#   data/real/ff48/window_<NN>/seed_00/{Y.npy,Y_test.npy,...}
# and results land at
#   results/real/ff48/window_<NN>/seed_00/<method>/...
# So we override --data-root and --results-root accordingly, and use the
# real-data config manifest produced by build_real_data_splits.py.

$PYTHON scripts/run_inference_single.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --task-manifest results/task_manifests/real_data.json \
    --config-manifest data/real/ff48/configs/config_manifest_real.json \
    --data-root data \
    --results-root results \
    --log-level INFO

echo "=== Task ${SLURM_ARRAY_TASK_ID} finished ==="
