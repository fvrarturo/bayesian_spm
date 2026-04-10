# Running DLO (Option Pricing) on MIT Engaging Cluster

This guide adapts the MIT Engaging workflow for the **Deep Learning for Option Pricing** project. Use it to upload the project **(including venv if you want)**, run the pipeline on the cluster, and download results.

**Export “all files including venv”:** use **Option A** in Step 1 (create archive **without** `--exclude='venv'`), then upload and unpack. If the job fails with Python/env errors on the cluster (macOS venv vs Linux), use **Option B** (exclude venv and recreate it on the cluster in Step 4).

---

## Login and hosts

- **SSH (from your Mac):** Use your Kerberos username. The cluster docs mention:
  - `ssh favara@eofe7.mit.edu`
  - Or you may land on a login node like `orcd-login001.mit.edu` / `orcd-login002.mit.edu`
- **Current command:** Check **Open OnDemand → Clusters → SSH Connection** for the exact `ssh yourKerberos@host` command.
- **VPN:** Connect to MIT VPN before `ssh` / `scp`.

---

## Step 1: Create archive on your Mac

From your **Mac**, in the directory that **contains** the `DLO` folder (e.g. `Desktop`):

```bash
cd ~/Desktop
```

### Option A: Include venv (larger; venv may not work on Linux)

```bash
tar -czf DLO_project.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='outputs/logs/*.log' \
  DLO
```

This includes `DLO/venv`. Note: a venv created on macOS often fails on the cluster (different OS, paths). If the job fails with Python/env errors, use Option B and recreate venv on the cluster.

### Option B: Exclude venv (recommended; recreate on cluster)

```bash
tar -czf DLO_project.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='venv' \
  --exclude='outputs/logs/*.log' \
  DLO
```

Then recreate the environment on the cluster (Step 4).

### Optional: script to create the archive

From project root (`DLO/`), you can run:

```bash
cd /Users/arturofavara/Desktop/DLO
bash _info/cluster/create_dlo_archive.sh
```

This creates `~/Desktop/DLO_project.tar.gz` (see script for include/exclude options).

---

## Step 2: Upload archive to the cluster

From your **Mac** (replace host if Open OnDemand gives a different one):

```bash
cd ~/Desktop
scp DLO_project.tar.gz favara@eofe7.mit.edu:~
```

If that fails (host key / connection):

- Connect to MIT VPN.
- Try: `ssh favara@eofe7.mit.edu` once to accept the host key, then run `scp` again.
- Or use the host from your cluster docs (e.g. `orcd-login001.mit.edu`):  
  `scp DLO_project.tar.gz favara@orcd-login001.mit.edu:~`

**Alternative:** Open OnDemand → **Files → Home Directory** and upload `DLO_project.tar.gz` there.

---

## Step 3: SSH into the cluster and unpack

On your **Mac**:

```bash
ssh favara@eofe7.mit.edu
```

(Or the SSH command from Open OnDemand.)

On the **cluster**:

```bash
cd ~
tar -xzf DLO_project.tar.gz
cd DLO
ls -la
```

You may see warnings like `tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'`. **These are harmless** — the archive still extracts correctly; Linux tar doesn’t use macOS extended attributes. You can ignore them.

You should see: `config.yaml`, `requirements.txt`, `src/`, `scripts/`, `data/`, `outputs/`, etc. If you used Option A, `venv/` will be there too.

---

## Step 4: Python environment on the cluster

### If you excluded venv (Option B) or the included venv fails

Recreate the environment on the cluster:

```bash
cd ~/DLO

# Try module first (if your cluster has it)
module load python/3.11 2>/dev/null || true

# Create venv (use same python that will run the job)
python3 -m venv venv
# If module load didn't work, use system Python 3.11 if available:
# /usr/bin/python3.11 -m venv venv

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check:

```bash
python -c "import pandas, numpy, scipy, yaml; print('OK')"
```

*(From CLUSTER_EXECUTION_SUMMARY, this cluster may not have `module load python/3.11`; in that case use `/usr/bin/python3.11 -m venv venv` and no `module load`.)*

### If you included venv (Option A)

```bash
cd ~/DLO
source venv/bin/activate
python -c "import pandas, numpy, scipy, yaml; print('OK')"
```

If you see import or path errors, delete `venv` and follow Option B above.

---

## Step 5: Raw data on the cluster

Your large OptionMetrics file must be on the cluster in one of these places:

- `~/DLO/data/raw/` (e.g. `options_data1.gz`), or  
- `~/DLO/` (project root).

If it was **not** in the archive (e.g. too large), upload it separately from your Mac:

```bash
scp /path/on/mac/to/options_data.csv.gz favara@eofe7.mit.edu:~/DLO/data/raw/
# or
scp /path/on/mac/to/options_data.csv.gz favara@eofe7.mit.edu:~/DLO/
```

---

## Step 6: Run the pipeline via SLURM

Long/compute-heavy work must run under SLURM, not on the login node.

From `~/DLO` on the cluster:

```bash
cd ~/DLO
sbatch _info/cluster/run_dlo_pipeline.slurm
```

This runs, in one job:

1. `scripts/run_cleaning.py`
2. `scripts/run_descriptive.py`
3. `scripts/run_baseline_bs.py`

Monitor:

```bash
squeue -u $USER
tail -f dlo_pipeline.out
tail -f dlo_pipeline.err
```

When the job is done, check:

```bash
ls -la data/processed/
ls -la outputs/figures/
ls -la outputs/tables/
```

---

## Step 7: Download results to your Mac

From your **Mac** (not on the cluster):

```bash
# Processed data (cleaned parquet)
scp -r favara@eofe7.mit.edu:~/DLO/data/processed ~/Desktop/DLO/data/

# Outputs (figures, tables, logs)
scp -r favara@eofe7.mit.edu:~/DLO/outputs/figures ~/Desktop/DLO/outputs/
scp -r favara@eofe7.mit.edu:~/DLO/outputs/tables ~/Desktop/DLO/outputs/
scp -r favara@eofe7.mit.edu:~/DLO/outputs/logs ~/Desktop/DLO/outputs/
```

Use the same host as in Step 2 (e.g. `eofe7.mit.edu` or `orcd-login001.mit.edu`).

---

## Quick reference

| Step | Where | Command |
|------|--------|--------|
| Archive | Mac | `cd ~/Desktop && tar -czf DLO_project.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='venv' DLO` |
| Upload | Mac | `scp DLO_project.tar.gz favara@eofe7.mit.edu:~` |
| Login | Mac | `ssh favara@eofe7.mit.edu` |
| Unpack | Cluster | `cd ~ && tar -xzf DLO_project.tar.gz && cd DLO` |
| Venv | Cluster | `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` |
| Run | Cluster | `cd ~/DLO && sbatch _info/cluster/run_dlo_pipeline.slurm` |
| Monitor | Cluster | `squeue -u $USER` and `tail -f dlo_pipeline.out` |
| Download | Mac | `scp -r favara@eofe7.mit.edu:~/DLO/data/processed ~/Desktop/DLO/data/` (and outputs as above) |

---

## Troubleshooting

- **“Host key verification failed” / “Connection closed”:** Use MIT VPN and the correct SSH host from Open OnDemand; run `ssh favara@...` once to accept the key.
- **`module load python/3.11` not found:** Use system Python, e.g. `/usr/bin/python3.11 -m venv venv`, and in the SLURM script use that interpreter or `source venv/bin/activate` without `module load`.
- **Job fails with Python/import errors:** Recreate venv on the cluster (don’t rely on the Mac venv).
- **Raw file not found:** Ensure the CSV.gz is in `~/DLO/data/raw/` or `~/DLO/` and that `config.yaml` and `src/data/loader.py` look in those locations.
- **Out of memory:** Increase `#SBATCH --mem=` in `run_dlo_pipeline.slurm` (e.g. 32G or 64G for very large files).
- **Partition time limit:** If the job is killed, reduce `#SBATCH -t` or split into multiple jobs (e.g. cleaning only, then descriptive, then baseline).
