# Sparse Bayesian Precision Matrix Estimation

**6.7830 Final Project — Spring 2026**

Nick Bernardini, Federico V. Cortesi, Arturo Favara — MIT

## Overview

This project compares MCMC (NUTS) and Variational Inference (ADVI) for sparse Bayesian precision matrix estimation using the **graphical horseshoe** prior (Li, Craig, and Bhadra, 2019), applied to equity return data. We systematically vary the concentration ratio p/T to identify regimes where (a) Bayesian regularization adds value over frequentist shrinkage, and (b) ADVI's mean-field approximation distorts the horseshoe's bimodal shrinkage profile.

## Setup

```bash
conda env create -f environment.yml
conda activate ggm_horseshoe
```

## Project Structure

```
data/           — raw, processed, and synthetic data
src/            — all source code (models, inference, benchmarks, evaluation, portfolio, utils)
notebooks/      — exploratory analysis
scripts/        — experiment runners and SLURM jobs
tests/          — unit tests
results/        — saved posteriors and metric tables
figures/        — publication-quality plots
paper/          — LaTeX source for final report
_info/          — project planning documents
```

## Running Experiments

```bash
python scripts/run_experiment.py --synthetic --p 50 --T 250 --method nuts
python scripts/run_experiment.py --synthetic --p 50 --T 250 --method advi --guide mean_field
```

## Tests

```bash
pytest tests/
```
