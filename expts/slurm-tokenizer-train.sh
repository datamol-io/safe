#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=train-safe-tokenizer
#SBATCH --output=/home/emmanuel/safe/expts/output/job_%x_%a.out
#SBATCH --error=/home/emmanuel/safe/expts/output/job_%x_%a.out
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=1:00:00

set -ex
# The below env variables can eventually help setting up your workload.
# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
source activate safe

TOKENIZER_TYPE="bpe"
DATASET="$1" # "/home/emmanuel/safe/expts/tmp_data/proc_data"
#DATASET="/home/emmanuel/safe/expts/notebook/tmp_data/proc_data"
OUTPUT="/home/emmanuel/safe/expts/tokenizer/tokenizer-custom.json"
VOCAB_SIZE="5000"
TEXT_COLUMN="input"
BATCH_SIZE="10000"

python scripts/tokenizer_trainer.py --tokenizer_type $TOKENIZER_TYPE \
                                    --dataset $DATASET --text_column $TEXT_COLUMN \
                                    --vocab_size $VOCAB_SIZE --batch_size $BATCH_SIZE \
                                    --all_split True --outfile $OUTPUT