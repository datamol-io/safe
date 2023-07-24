from typing import Optional
from typing import Literal

import os
import sys
import typer
import torch
import logging
from transformers import (
    AutoConfig,
    set_seed,
)

from safe.trainer.model import SAFEDoubleHeadsModel
from safe.tokenizer import SAFETokenizer
from safe.trainer.data_utils import get_dataset


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    fp16: Optional[bool] = typer.Option(False, "--fp16", help="whether to train using fp16"),
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
    wandb_watch: Literal["gradients", "all", None] = typer.Option(
        None,
        "--wandb-project",
        prompt=True,
        show_choices=True,
        help="whern of the wandb project to use to log the SAFE model parameter",
    ),
    # whether to resume training from a checkpoint
    resume_from_checkpoint: Optional[str] = typer.Option(
        None,
        "--resume-from-checkpoint",
        help="Path to a checkpoint from which we can resume experiments",
    ),
    eval_every: Optional[int] = typer.Option(
        100, "--eval-every", help="Number of steps between every evaluation"
    ),
    save_every: Optional[int] = typer.Option(
        100, "--save-every", help="Number of steps between every model saving"
    ),
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    # Only overwrite environ if wandb param passed
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

    # load dataset
    dataset = get_dataset(dataset, streaming=True)

    # load tokenizer and model
    tokenizer = SAFETokenizer.load(tokenizer)
    if config is None:
        config = os.path.join(CURRENT_DIR, "data/default_config.json")
    config = AutoConfig.from_pretrained(config, cache_dir=cache_dir)
    if model_path is not None is not None:
        model = SAFEDoubleHeadsModel.from_pretrained(model_path, config=config)
    else:
        model = SAFEDoubleHeadsModel(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if is_tokenized:
        pass

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        logger.info(f"Restarting from {checkpoint_name}")

    if not ddp and n_gpu > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    #     args=,
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )
    # model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    # ).__get__(model, type(model))
    # # with torch.autocast("cuda"):
    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()

    # print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    app()
