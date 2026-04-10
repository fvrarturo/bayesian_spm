# ORCD / MIT Engaging — Command reference

Quick reference for terminal commands. Run **Mac** commands on your laptop; run **Cluster** commands after SSH into the cluster. Replace `favara` with your username and `eofe7.mit.edu` with your login host if different (check Open OnDemand → SSH Connection).

---

## Connection

| Action | Where | Command |
|--------|--------|--------|
| **Login to cluster** | Mac | `ssh favara@eofe7.mit.edu` |
| **Logout** | Cluster | `exit` |
| **VPN** | Mac | Connect to MIT VPN before SSH/rsync/scp |

---

## Jobs (SLURM)

| Action | Where | Command |
|--------|--------|--------|
| **Submit explore (columns/dtypes)** | Cluster | `cd ~/DLO && sbatch slurms/run_explore.slurm` |
| **Submit cleaning job** | Cluster | `cd ~/DLO && sbatch slurms/run_cleaning.slurm` |
| **Submit full pipeline** | Cluster | `cd ~/DLO && sbatch _info/cluster/run_dlo_pipeline.slurm` |
| **List your jobs** | Cluster | `squeue -u $USER` or `squeue -u favara` |
| **Cancel a job** | Cluster | `scancel <JOB_ID>` (e.g. `scancel 9209713`) |
| **Cancel all your jobs** | Cluster | `scancel -u favara` |
| **Watch stdout live** | Cluster | `tail -f ~/DLO/dlo_cleaning.out` or `tail -f ~/DLO/dlo_explore.out` |
| **Watch stderr live** | Cluster | `tail -f ~/DLO/dlo_cleaning.err` or `tail -f ~/DLO/dlo_explore.err` |
| **Last lines of error file** | Cluster | `tail -100 ~/DLO/dlo_cleaning.err` |

tail -f ~/DLO/dlo_cleaning.out
---

## Sync project (rsync) — skip heavy data

| Action | Where | Command |
|--------|--------|--------|
| **Mac → cluster** (no data/venv/info/papers)** | Mac | rsync -avz --exclude='.*' --exclude='data/' --exclude='_info/' --exclude='venv/'  /Users/arturofavara/Desktop/SBI_rBergomi/ favara@eofe7.mit.edu:~/SBI_rBergomi/ |
| **Cluster → Mac** (same excludes) | Mac | `rsync -avz --exclude='data/raw/' --exclude='data/processed/' --exclude='venv/' --exclude='__pycache__/' favara@eofe7.mit.edu:~/DLO/ /Users/arturofavara/Desktop/DLO/` |
| **Upload one file** | Mac | `scp slurms/run_optionprice_initial_plots.slurm favara@eofe7.mit.edu:~/DLO/slurms/` |


---

## Download files from cluster to Mac

| Action | Where | Command |
|--------|--------|--------|
| **Job logs only** | Mac | `rsync -avz favara@eofe7.mit.edu:~/DLO/dlo_cleaning.out favara@eofe7.mit.edu:~/DLO/dlo_cleaning.err /Users/arturofavara/Desktop/DLO/` |
| **Outputs (figures, tables, logs)** | Mac | `rsync -avz favara@eofe7.mit.edu:~/DLO/outputs/ /Users/arturofavara/Desktop/DLO/outputs/` |
| **Processed data** | Mac | `rsync -avz favara@eofe7.mit.edu:~/DLO/data/processed/ /Users/arturofavara/Desktop/DLO/data/processed/` |
| **Single file** | Mac | `scp favara@eofe7.mit.edu:~/DLO/dlo_cleaning.err /Users/arturofavara/Desktop/DLO/` |

---

## Environment on cluster (interactive)

| Action | Where | Command |
|--------|--------|--------|
| **Activate conda env `dlo`** | Cluster | `source activate dlo` |
| **Run cleaning interactively** | Cluster | `cd ~/DLO && source activate dlo && python scripts/run_cleaning.py` |
| **Check Python** | Cluster | `which python && python --version` |
| **List conda envs** | Cluster | `conda env list` |

---

## One-off upload (scp)

| Action | Where | Command |
|--------|--------|--------|
| **Upload one file** | Mac | `scp /path/to/local/file favara@eofe7.mit.edu:~/DLO/data/raw/` |
| **Upload tarball** | Mac | `scp ~/Desktop/DLO_project.tar.gz favara@eofe7.mit.edu:~` |

---

## Paths on cluster

| What | Path |
|------|------|
| **Project root** | `~/DLO` or `/orcd/home/002/favara/DLO` |
| **Job stdout/stderr** | `~/DLO/dlo_cleaning.out`, `~/DLO/dlo_cleaning.err` |
| **Raw data** | `~/DLO/data/raw/` |
| **Processed data** | `~/DLO/data/processed/` |
| **Outputs** | `~/DLO/outputs/figures/`, `~/DLO/outputs/tables/`, `~/DLO/outputs/logs/` |
