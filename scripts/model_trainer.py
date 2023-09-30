#!/usr/bin/env python
import transformers
from safe.trainer.cli import ModelArguments
from safe.trainer.cli import DataArguments
from safe.trainer.cli import TrainingArguments
from safe.trainer.cli import train

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
train(model_args, data_args, training_args)
