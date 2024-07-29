import math
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

import datasets
import evaluate
import torch
import transformers
from loguru import logger
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import log_levels as LOG_LEVELS

import safe
from safe.tokenizer import SAFETokenizer
from safe.trainer.collator import SAFECollator
from safe.trainer.data_utils import get_dataset
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.trainer.trainer_utils import SAFETrainer

CURRENT_DIR = os.path.join(safe.__path__[0], "trainer")


@dataclass
class ModelArguments:
    model_path: str = field(
        default=None,
        metadata={
            "help": "Optional model path or model name to use as a starting point for the safe model"
        },
    )
    config: Optional[str] = field(
        default=None, metadata={"help": "Path to the default config file to use for the safe model"}
    )

    tokenizer: str = (
        field(
            default=None,
            metadata={"help": "Path to the trained tokenizer to use to build a safe model"},
        ),
    )
    num_labels: Optional[int] = field(
        default=None, metadata={"help": "Optional number of labels for the descriptors"}
    )
    include_descriptors: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train with descriptors if they are available or Not"},
    )
    prop_loss_coeff: Optional[float] = field(
        default=1e-2, metadata={"help": "coefficient for the propery loss"}
    )
    model_hub_name: Optional[str] = field(
        default="maclandrol/safe-gpt2",
        metadata={"help": "Name of the model when uploading to huggingface"},
    )

    wandb_project: Optional[str] = field(
        default="safe-gpt2",
        metadata={"help": "Name of the wandb project to use to log the SAFE model parameter"},
    )
    wandb_watch: Optional[Literal["gradients", "all"]] = field(
        default=None, metadata={"help": "Whether to watch the wandb models or not"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption. Only valid when loading a pretrained model"
            )
        },
    )
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated) up to that value."
        },
    )


@dataclass
class DataArguments:
    dataset: str = field(
        default=None,
        metadata={"help": "Path to the preprocessed dataset to use for the safe model building"},
    )
    is_tokenized: Optional[bool] = field(
        default=False,
        metadata={"help": "whether the dataset submitted as input is already tokenized or not"},
    )

    streaming: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a streaming dataset or not"}
    )

    text_column: Optional[str] = field(
        default="inputs", metadata={"help": "Column containing text data to process."}
    )

    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of training sample to use."}
    )

    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of evaluation sample to use."}
    )

    train_split: Optional[str] = field(
        default="train",
        metadata={
            "help": "Training splits to use. You can use train+validation for example to include both train and validation split in the training"
        },
    )

    property_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Column containing the descriptors information. Default to None to use `mc_labels`"
        },
    )


def train(model_args, data_args, training_args):
    """Train a new model from scratch"""
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    if log_level is None or log_level == "passive":
        log_level = "info"

    _LOG_LEVEL = {v: k for k, v in LOG_LEVELS.items()}
    logger.remove()
    logger.add(sys.stderr, level=_LOG_LEVEL.get(log_level, log_level).upper())
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Training/evaluation parameters: {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Check if parameter passed or if set within environ
    # Only overwrite environ if wandb param passed
    wandb_run_name = f"safe-model-{uuid.uuid4().hex[:8]}"
    if model_args.wandb_project:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project
        training_args.report_to = ["wandb"]
    if model_args.wandb_watch:
        os.environ["WANDB_WATCH"] = model_args.wandb_watch
        if model_args.wandb_watch == "all":
            os.environ["WANDB_LOG_MODEL"] = "end"

    training_args.run_name = wandb_run_name
    training_args.remove_unused_columns = False
    # load tokenizer and model

    set_seed(training_args.seed)
    # load the tokenizer
    if model_args.tokenizer.endswith(".json"):
        tokenizer = SAFETokenizer.load(model_args.tokenizer)
    else:
        try:
            tokenizer = SAFETokenizer.load(model_args.tokenizer)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer)

    # load dataset
    with training_args.main_process_first():
        # if the dataset is streaming we tokenize on the fly, it would be faster
        dataset = get_dataset(
            data_args.dataset,
            tokenizer=(None if data_args.is_tokenized or data_args.streaming else tokenizer),
            streaming=data_args.streaming,
            tokenize_column=data_args.text_column,
            max_length=model_args.model_max_length,
            property_column=data_args.property_column,
        )

        if data_args.max_train_samples is not None:
            dataset["train"] = dataset["train"].take(data_args.max_train_samples)
        if data_args.max_eval_samples is not None:
            for k in dataset:
                if k != "train":
                    dataset[k] = dataset[k].take(data_args.max_eval_samples)

    eval_dataset_key_name = "validation" if "validation" in dataset else "test"

    train_dataset = dataset["train"]
    if data_args.train_split is not None:
        train_dataset = datasets.concatenate_datasets(
            [dataset[x] for x in data_args.train_split.split("+")]
        )
        if eval_dataset_key_name in data_args.train_split.split("+"):
            eval_dataset_key_name = None

    data_collator = SAFECollator(
        tokenizer=tokenizer,
        input_key=data_args.text_column,
        max_length=model_args.model_max_length,
        include_descriptors=model_args.include_descriptors,
        property_key="mc_labels",
    )
    pretrained_tokenizer = data_collator.get_tokenizer()
    config = model_args.config

    if config is None:
        config = os.path.join(CURRENT_DIR, "configs/default_config.json")
    config = AutoConfig.from_pretrained(config, cache_dir=model_args.cache_dir)

    if model_args.num_labels is not None:
        config.num_labels = int(model_args.num_labels)

    config.vocab_size = len(tokenizer)
    if model_args.model_max_length is not None:
        config.max_position_embeddings = model_args.model_max_length
    try:
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
    except:
        config.bos_token_id = pretrained_tokenizer.bos_token_id
        config.eos_token_id = pretrained_tokenizer.eos_token_id
        config.pad_token_id = pretrained_tokenizer.pad_token_id

    if model_args.model_path is not None:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = SAFEDoubleHeadsModel.from_pretrained(
            model_args.model_path,
            config=config,
            cache_dir=model_args.cache_dir,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            torch_dtype=torch_dtype,
        )

    else:
        model = SAFEDoubleHeadsModel(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    def preprocess_logits_for_metrics(logits, labels):
        prop_logits = None
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            if len(logits) > 1:
                # we could have the loss twice
                base_ind = 0
                if logits[base_ind].ndim < 2:
                    base_ind = 1
            label_logits = logits[base_ind].argmax(dim=-1)
            if len(logits) > base_ind + 1:
                prop_logits = logits[base_ind + 1]
        else:
            label_logits = logits.argmax(dim=-1)

        if prop_logits is not None:
            return label_logits, prop_logits
        return label_logits

    accuracy_metric = evaluate.load("accuracy")
    mse_metric = evaluate.load("mse")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        mc_preds = None
        mc_labels = None
        if isinstance(preds, tuple):
            preds, mc_preds = preds
        if isinstance(labels, tuple):
            labels, mc_labels = labels
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        results = accuracy_metric.compute(predictions=preds, references=labels)
        if mc_preds is not None and mc_labels is not None:
            results_mse = mse_metric.compute(
                predictions=mc_preds.reshape(-1), references=mc_labels.reshape(-1)
            )
            results.update(results_mse)
        return results

    if model_args.include_descriptors:
        training_args.label_names = ["labels", "mc_labels"]

    if training_args.label_names is None:
        training_args.label_names = ["labels"]
    # update dispatch_batches in accelerator
    training_args.accelerator_config.dispatch_batches = data_args.streaming is not True

    trainer = SAFETrainer(
        model=model,
        tokenizer=None,  # we don't deal with the tokenizer at all, https://github.com/huggingface/tokenizers/issues/581 -_-
        train_dataset=train_dataset.shuffle(seed=(training_args.seed or 42)),
        eval_dataset=dataset.get(eval_dataset_key_name, None),
        args=training_args,
        prop_loss_coeff=model_args.prop_loss_coeff,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        data_collator=data_collator,
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval else None
        ),
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        try:
            # we were unable to save the model because of the tokenizer
            trainer.save_model()  # Saves the tokenizer too for easy upload
        except:
            model.save_pretrained(os.path.join(training_args.output_dir, "safe-model"))

        if training_args.push_to_hub and model_args.model_hub_name:
            model.push_to_hub(model_args.model_hub_name, private=True, safe_serialization=True)

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save(os.path.join(training_args.output_dir, "tokenizer.json"))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()
        try:
            perplexity = math.exp(results["eval_loss"])
        except Exception as e:
            logger.error(e)
            perplexity = float("inf")
        results.update({"perplexity": perplexity})
        if trainer.is_world_process_zero():
            trainer.log_metrics("eval", results)
            trainer.save_metrics("eval", results)

    kwargs = {"finetuned_from": model_args.model_path, "tasks": "text-generation"}
    kwargs["dataset_tags"] = data_args.dataset
    kwargs["dataset"] = data_args.dataset
    kwargs["tags"] = ["safe", "datamol-io", "molecule-design", "smiles"]

    if training_args.push_to_hub:
        kwargs["private"] = True
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
