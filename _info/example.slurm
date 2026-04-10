#!/bin/bash
#SBATCH -J dlo_clean_panel
#SBATCH -o outputs/logs/dlo_clean_panel_%j.out
#SBATCH -e outputs/logs/dlo_clean_panel_%j.err
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=386G
#SBATCH -p mit_normal
# mit_normal QOS limit: mem=386G per user (sacctmgr show qos). Use max to avoid OOM on ~45M rows.

# Build cleaned option_prices panel: merge forwards, normalise strikes,
# apply filters, compute self-derived IVs and Greeks.
# Processes per-secid for bounded memory.

cd $SLURM_SUBMIT_DIR
PYTHON="${HOME}/.conda/envs/dlo/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load anaconda3/2023.07 2>/dev/null || true
  source activate dlo 2>/dev/null || true
  PYTHON=python
fi

set -e
echo "=== Building cleaned option_prices panel ==="
$PYTHON scripts/run_build_clean_panel.py
echo "=== Done. Output in outputs/data/ ==="
