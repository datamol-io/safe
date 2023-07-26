from typing import Optional
from typing import Literal

import math
import os
import uuid
import safe
import transformers
from dataclasses import dataclass, field
from loguru import logger
from transformers import AutoConfig
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.tokenizer import SAFETokenizer
from safe.trainer.data_utils import get_dataset
from safe.trainer.collator import SAFECollator
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
    prop_loss_coeff: Optional[float] = field(
        default=1e-2, metadata={"help": "coefficient for the propery loss"}
    )
    wandb_project: Optional[str] = field(
        default="safe-gpt2",
        metadata={"help": "Name of the wandb project to use to log the SAFE model parameter"},
    )
    wandb_watch: Optional[Literal["gradients", "all"]] = field(
        default=None, metadata={"help": "Whether to watch the wandb models or not"}
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


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Cache dir to use to speed up downloading and loading data"}
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def train():
    """Train a new model from scratch"""
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check if parameter passed or if set within environ
    # Only overwrite environ if wandb param passed
    wandb_run_name = f"safe-model-{uuid.uuid4().hex[:8]}"
    if model_args.wandb_project:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project
    if model_args.wandb_watch:
        os.environ["WANDB_WATCH"] = model_args.wandb_watch
        if model_args.wandb_watch == "all":
            os.environ["WANDB_LOG_MODEL"] = "end"

    training_args.run_name = wandb_run_name
    training_args.remove_unused_columns = False
    # load tokenizer and model
    tokenizer = SAFETokenizer.load(model_args.tokenizer)

    # load dataset
    with training_args.main_process_first():
        dataset = get_dataset(
            data_args.dataset,
            tokenizer=(None if data_args.is_tokenized else tokenizer),
            streaming=data_args.streaming,
            max_length=training_args.model_max_length,
        )

    data_collator = SAFECollator(tokenizer=tokenizer, max_length=training_args.model_max_length)
    pretrained_tokenizer = data_collator.get_tokenizer()
    config = model_args.config

    if config is None:
        config = os.path.join(CURRENT_DIR, "configs/default_config.json")
    config = AutoConfig.from_pretrained(config, cache_dir=training_args.cache_dir)

    if model_args.num_labels is not None:
        config.num_labels = int(model_args.num_labels)

    config.vocab_size = len(tokenizer)
    if training_args.model_max_length is not None:
        config.max_position_embeddings = training_args.model_max_length

    try:
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
    except:
        pretrained_tokenizer = data_collator.get_tokenizer()
        config.bos_token_id = pretrained_tokenizer.bos_token_id
        config.eos_token_id = pretrained_tokenizer.eos_token_id
        config.pad_token_id = pretrained_tokenizer.pad_token_id

    if model_args.model_path is not None:
        model = SAFEDoubleHeadsModel.from_pretrained(
            model_args.model_path,
            config=config,
            cache_dir=training_args.cache_dir,
            low_cpu_mem_usage=True,
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

    trainer = SAFETrainer(
        model=model,
        tokenizer=data_collator.get_tokenizer(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        data_collator=data_collator,
        prop_loss_coeff=model_args.prop_loss_coeff,
    )
    trainer.train()
    trainer.save_state()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save(os.path.join(training_args.output_dir, "tokenizer.json"))

    # Evaluation
    results = {}
    logger.info("*** Evaluate ***")
    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    results.update(result)


if __name__ == "__main__":
    train()
