# Model-training CLI

`safe-train` trains or fine-tunes a SAFE generative model using the supported Transformers 4.57 stack.

The tokenizer and dataset are required for training. The output directory is managed by the Transformers trainer:

```bash
safe-train \
  --tokenizer path/to/tokenizer.json \
  --dataset path/to/dataset \
  --output_dir path/to/output \
  --do_train \
  --max_steps 1000
```

To start from an existing checkpoint, add `--model_path`. To use a custom model configuration, add `--config`; otherwise SAFE uses its packaged GPT-2 configuration.

Common options include:

| Option | Purpose |
| --- | --- |
| `--tokenizer` | Required tokenizer JSON file, local tokenizer directory, or compatible Hub identifier. |
| `--dataset` | Required local Datasets directory or Hugging Face dataset identifier. |
| `--model_path` | Optional checkpoint used to initialize the model. |
| `--config` | Optional Transformers configuration; defaults to SAFE's packaged configuration. |
| `--streaming` | Stream the dataset instead of loading it eagerly. |
| `--text_column` | Dataset column containing SAFE strings; defaults to `inputs`. |
| `--include_descriptors` | Enable the auxiliary property head. |
| `--property_column` | Dataset column containing descriptor targets. |
| `--model_max_length` | Token sequence limit; defaults to 1024. |
| `--wandb_project` | Weights & Biases project name. Set an empty value to disable W&B reporting. |
| `--do_train`, `--do_eval`, `--do_predict` | Select the trainer operations to execute. |

All standard `transformers.TrainingArguments` are also available. Run the installed command to see the exact options for the pinned Transformers release:

```bash
safe-train --help
```

Transformers 5 is not supported because it removed the constrained-generation API used elsewhere in SAFE. See [Migrating to SAFE 1.0](migration.md) for the compatibility rationale.
