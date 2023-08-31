#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=jupyter
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=1-12:00

set -ex
# The below env variables can eventually help setting up your workload.
# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
source activate safe
python scripts/build_dataset.py