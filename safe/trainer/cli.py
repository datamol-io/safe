from typing import Optional

import math
import os
import sys
import typer
import uuid
import click
import torch
import logging
import transformers
from transformers import (
    AutoConfig,
    Trainer,
    set_seed,
)
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.tokenizer import SAFETokenizer
from safe.trainer.data_utils import get_dataset


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGGING_STEPS = 100
logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def train(
    config: Optional[str] = typer.Option(
        ..., "--config", "-c", help="Path to teh default config file to use for the safe model"
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Optional model path or model name to use as a starting point for the safe model",
    ),
    dataset: Optional[str] = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to the preprocessed dataset to use for the safe model building",
    ),
    tokenizer: Optional[str] = typer.Option(
        ..., "--tokenizer", "-t", help="Path to the trained tokenizer to use to build a safe model"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output dir to use for training and artifact generation of the safe model",
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help="Cache dir to use to speed up downloading and loading data"
    ),
    # training hyperparams
    batch_size: Optional[int] = typer.Option(
        64, "--batch_size", "-b", help="Batch size to use for the model training"
    ),
    num_epochs: Optional[int] = typer.Option(
        10, "--num_epochs", "-N", help="number of training epochs"
    ),
    learning_rate: Optional[float] = typer.Option(
        3e-4, "--lr", help="learning rate to use for training"
    ),
    is_tokenized: Optional[bool] = typer.Option(
        False,
        "--is-tokenized",
        help="whether the dataset submitted as input is already tokenized or not",
    ),
    gradient_accumulation_steps: Optional[int] = typer.Option(
        1, "--gradient-accumulation-steps", help="gradient accumulation steps to use for training"
    ),
    seed: Optional[int] = typer.Option(
        ..., "--seed", help="Optional seed to set for reproducibility of the experiments"
    ),
    # llm hyperparams
    dual_head: Optional[bool] = typer.Option(
        False, "--dual-head", help="whether to use a dual head loss"
    ),
    dtype: Optional[str] = typer.Option(
        None, "--dtype", help="torch datatype to use and train the model in"
    ),
    group_by_length: Optional[bool] = typer.Option(
        False,
        "--group-by-length",
        help="whether to group the dataset sequence by length for the batch",
    ),
    # wandb params
    wandb_project: Optional[bool] = typer.Option(
        "safe-gpt2",
        "--wandb-project",
        help="Name of the wandb project to use to log the SAFE model parameter",
    ),
    wandb_watch: Optional[str] = typer.Option(
        None,
        "--wandb-watch",
        help="Whether to watch the wandb models or not",
    ),
    # whether to resume training from a checkpoint
    resume_from_checkpoint: Optional[str] = typer.Option(
        None,
        "--resume-from-checkpoint",
        help="Path to a checkpoint from which we can resume experiments",
    ),
    eval_every: Optional[int] = typer.Option(
        1000, "--eval-every", help="Number of steps between every evaluation"
    ),
    save_every: Optional[int] = typer.Option(
        1000, "--save-every", help="Number of steps between every model saving"
    ),
    warmup_steps: Optional[int] = typer.Option(
        None,
        "--warmup-steps",
        help="Number of steps used for a linear warmup from 0 to learning_rate",
    ),
    num_workers: Optional[int] = typer.Option(
        0, "--num-workers", help="Number of workers to use for dataloading"
    ),
    max_steps: Optional[int] = typer.Option(-1, "--max-steps", help="Maximum number of steps"),
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if dtype not in ["auto", "bfloat16", "float16", "float32", None]:
        raise ValueError(
            f"Invalid dtype {dtype}", "Valid values are: auto, bfloat16, float16, float32, None"
        )

    if wandb_watch not in ["gradients", "all", None]:
        raise ValueError(
            f"Invalid wandb_watch {wandb_watch}", "Valid values are: gradients, all, None"
        )
    # Check if parameter passed or if set within environ
    # Only overwrite environ if wandb param passed
    wandb_run_name = f"safe-model-{uuid.uuid4().hex[:8]}"
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_watch:
        os.environ["WANDB_WATCH"] = wandb_watch
        if wandb_watch == "all":
            os.environ["WANDB_LOG_MODEL"] = True

    # set seed
    set_seed(seed)

    # configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    # load tokenizer and model
    tokenizer = SAFETokenizer.load(tokenizer)
    if config is None:
        config = os.path.join(CURRENT_DIR, "configs/default_config.json")
    config = AutoConfig.from_pretrained(config, cache_dir=cache_dir)

    config.vocab_size = len(tokenizer)
    torch_dtype = dtype if dtype in ["auto", None] else getattr(torch, dtype)

    if model_path is not None:
        model = SAFEDoubleHeadsModel.from_pretrained(
            model_path,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
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

    model.to(device)

    training_args = (
        transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=(dtype == "float16"),
            logging_steps=LOGGING_STEPS,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_every,
            save_steps=save_every,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if wandb_project is not None else None,
            run_name=wandb_run_name if wandb_project is not None else None,
            dataloader_num_workers=num_workers,
            save_safetensors=True,
            torch_compile=(torch.__version__ >= "2" and sys.platform != "win32"),
            max_steps=max_steps,
        ),
    )

    # load dataset
    with training_args.main_process_first():
        dataset = get_dataset(
            dataset, tokenizer=(None if is_tokenized else tokenizer), streaming=True
        )

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        logger.info(f"Restarting from {checkpoint_name}")

    if not ddp and n_gpu > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0.0
    )

    data_collator = (
        transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        data_collator=data_collator,
        prediction_loss_only=True,
    )

    train_params = {}
    if model_path is not None and os.path.isdir(model_path):
        train_params["model_path"] = model_path

    if resume_from_checkpoint is not None:
        train_params["resume_from_checkpoint"] = resume_from_checkpoint

    trainer.train(**train_params)
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_master():
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    model.save_pretrained(output_dir)

    # Evaluation
    results = {}
    logger.info("*** Evaluate ***")

    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(output_dir, "eval_results_lm.txt")
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    results.update(result)


if __name__ == "__main__":
    app()
