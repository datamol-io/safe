#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=split-train-safe-tokenizer
#SBATCH --output=/home/emmanuel/safe/expts/output/job_split_%x_%a.out
#SBATCH --error=/home/emmanuel/safe/expts/output/job_split_%x_%a.out
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=48:00:00

set -ex
# The below env variables can eventually help setting up your workload.
# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
source activate safe

TOKENIZER_TYPE="bpe"
DEFAULT_DATASET="/storage/shared_data/cristian/preprocessed_zinc_unichem/train_filtered/"
DATASET="${1:-$DEFAULT_DATASET}"
#DATASET="/home/emmanuel/safe/expts/notebook/tmp_data/proc_data"
OUTPUT="/home/emmanuel/safe/expts/tokenizer/tokenizer-custom.json"
VOCAB_SIZE="10000"
TEXT_COLUMN="input"
BATCH_SIZE="1000"
N_EXAMPLES="500000000"

python scripts/tokenizer_trainer.py --tokenizer_type $TOKENIZER_TYPE \
                                    --dataset $DATASET --text_column $TEXT_COLUMN \
                                    --vocab_size $VOCAB_SIZE --batch_size $BATCH_SIZE \
                                    --outfile $OUTPUT --splitter 'safe' --n_examples $N_EXAMPLES --tokenizer_name "safe-custom"