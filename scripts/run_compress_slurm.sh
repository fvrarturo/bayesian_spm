#!/bin/bash
#SBATCH --job-name=compress_results
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=logs/compress_%j.out
#SBATCH --error=logs/compress_%j.err

# =====================================================================
# Compress legacy .npy posterior-sample arrays to zlib .npz, in parallel.
#
# Single node, 16 workers via multiprocessing.Pool.  With ~162 GB of
# .npy to compress and downcast float64->float32, expect 20-60 min wall
# depending on filesystem throughput.
#
# Safe under interruption: each file is atomic (`.npz.tmp` -> rename)
# and the .npy is only deleted AFTER the .npz lands on disk.  Re-running
# skips files that are already `.npz`.
#
# Dry-run mode: comment out the real invocation at the bottom and
# uncomment the --dry-run line.
# =====================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

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

# Keep BLAS single-threaded per worker — compression is not a BLAS op,
# and oversubscribing hurts.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

WORKERS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  workers=${WORKERS} ==="
echo "=== Python: $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])') ==="
echo "=== Disk before: $(df -h . | tail -1) ==="

# ---- Dry run (uncomment to preview) ---------------------------------
# $PYTHON scripts/compress_results.py --dry-run --quiet
# exit 0
# ---------------------------------------------------------------------

$PYTHON scripts/compress_results.py \
    --workers "${WORKERS}" \
    --quiet

echo "=== Disk after:  $(df -h . | tail -1) ==="
echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
