from typing import Optional, List

import tempfile
import os
import contextlib
import torch
import wandb
import fsspec

from transformers import PreTrainedModel, is_torch_available
from transformers.processing_utils import PushToHubMixin


def upload_to_wandb(
    model: PreTrainedModel,
    tokenizer,
    artifact_name: str,
    wandb_project_name: Optional[str] = "safe-models",
    artifact_type: str = "model",
    slicer: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    **init_args,
):
    """
    Uploads a model and tokenizer to a specified Weights and Biases (wandb) project.

    Args:
        model (PreTrainedModel): The model to be uploaded (instance of PreTrainedModel).
        tokenizer: The tokenizer associated with the model.
        artifact_name (str): The name of the wandb artifact to create.
        wandb_project_name (Optional[str]): The name of the wandb project. Defaults to 'safe-model'.
        artifact_type (str): The type of artifact (e.g., 'model'). Defaults to 'model'.
        slicer (Optional[str]): Optional metadata field that can store a slicing method.
        aliases (Optional[List[str]]): List of aliases to assign to this artifact version.
        **init_args: Additional arguments to pass into `wandb.init()`.
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Paths to save model and tokenizer
        model_path = tokenizer_path = tmpdirname
        architecture_file = os.path.join(tmpdirname, "architecture.txt")
        with fsspec.open(architecture_file, "w+") as f:
            f.write(str(model))

        model.save_pretrained(model_path)
        with contextlib.suppress(Exception):
            tokenizer.save_pretrained(tokenizer_path)
        tokenizer.save(os.path.join(tokenizer_path, "tokenizer.json"))

        info_dict = {"slicer": slicer}
        model_config = None
        if hasattr(model, "config") and model.config is not None:
            model_config = (
                model.config.to_dict() if not isinstance(model.config, dict) else model.config
            )
            info_dict.update(model_config)

        if hasattr(model, "peft_config") and model.peft_config is not None:
            info_dict.update({"peft_config": model.peft_config})

        with contextlib.suppress(Exception):
            info_dict["model/num_parameters"] = model.num_parameters()

        init_args.setdefault("config", info_dict)
        run = wandb.init(project=os.getenv("SAFE_WANDB_PROJECT", wandb_project_name), **init_args)

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata={
                "model_config": model_config,
                "num_parameters": info_dict.get("model/num_parameters"),
                "initial_model": True,
            },
        )

        # Add model and tokenizer directories to the artifact
        artifact.add_dir(tmpdirname)
        run.log_artifact(artifact, aliases=aliases)

        # Finish the wandb run
        run.finish()
