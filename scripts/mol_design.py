from typing import Optional
from typing_extensions import Annotated

import typer

import torch
import safe as sf
import datamol as dm
import numpy as np
import pandas as pd
import wandb


from tqdm.auto import tqdm
from enum import Enum
from safe import SAFEDesign
from tdc import Oracle
from tdc import Evaluator
from random import choices
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model


app = typer.Typer()


class DesignMode(str, Enum):
    denovo = "denovo"
    superstructure = "superstructure"
    decoration = "decoration"
    motif_extension = "motif_extension"
    morphing = "morphing"
    linker = "linker"


class DesignObjective(str, Enum):
    clogp = "clogp"
    qed = "qed"
    sas = "sa"
    drd2 = "drd2"


def train(
    ppo_config,
    generation_kwargs,
    model,
    tokenizer,
    oracle,
    prefix=None,
    n_episodes=100,
    batch_size=32,
):
    model_ref = create_reference_model(model)
    config = PPOConfig(**ppo_config)

    diversity_evaluator = Evaluator(name="Diversity")
    Evaluator(name="Validity")
    uniqueness_evaluator = Evaluator(name="Uniqueness")

    ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)
    if prefix is None:
        prefix = tokenizer.bos_token

    if isinstance(prefix, str):
        prefix = [prefix]

    if len(prefix) < batch_size:
        prefix = choices(prefix, k=batch_size)

    for _ in tqdm(range(n_episodes)):
        game_data = {}
        game_data["query"] = prefix
        batch = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(
            model.pretrained_model.device
        )
        query_tensor = batch["input_ids"]
        response_tensor = ppo_trainer.generate(
            list(query_tensor), return_prompt=False, **generation_kwargs
        )
        decoded_safe_mols = tokenizer.batch_decode(response_tensor, skip_special_tokens=True)

        decoded_smiles = [
            sf.decode(
                x,
                as_mol=False,
                fix=True,
                remove_added_hs=True,
                canonical=True,
                ignore_errors=True,
                remove_dummies=True,
            )
            for x in decoded_safe_mols
        ]

        game_data["response"] = decoded_safe_mols
        valid_position, valid_smiles = zip(
            *[(i, x) for i, x in enumerate(decoded_smiles) if x is not None]
        )
        batch_reward = oracle(list(valid_smiles))
        rewards = np.zeros(len(decoded_smiles))
        rewards[np.asarray(valid_position)] = batch_reward
        rewards = torch.from_numpy(rewards).to(device=model.pretrained_model.device)
        rewards = list(rewards)
        stats = ppo_trainer.step(list(query_tensor), list(response_tensor), rewards)
        stats["validity"] = len(valid_position) / batch_size
        stats["uniqueness"] = uniqueness_evaluator(list(valid_smiles))
        stats["diversity"] = diversity_evaluator(list(valid_smiles))
        ppo_trainer.log_stats(stats, game_data, rewards)
    return ppo_trainer, model


@app.command()
def sample(
    checkpoint: Annotated[Optional[str], typer.Option()] = None,
    n_samples: Annotated[int, typer.Option(default=100)] = 1000,
    n_trials: Annotated[int, typer.Option(default=1)] = 1,
    sanitize: Annotated[bool, typer.Option(default=True)] = True,
    allow_further_decomposition: Annotated[bool, typer.Option(default=False)] = False,
    mode: Annotated[DesignMode, typer.Option(case_sensitive=False)] = DesignMode.denovo,
    inputs: Annotated[str, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 42,
    outfile: Annotated[str, typer.Option(default=...)] = None,
):
    """Sample molecule using SAFEDesign"""

    designer = SAFEDesign.load_default(verbose=False, model_dir=checkpoint)
    generate_params = {
        "n_samples_per_trial": n_samples,
        "n_trials": n_trials,
        "sanitize": sanitize,
        "do_not_fragment_further": (not allow_further_decomposition),
        "random_seed": seed,
    }

    if mode.value == "denovo":
        generated = designer.de_novo_generation(
            n_samples_per_trial=n_samples, n_trials=n_trials, sanitize=sanitize
        )
    elif mode.value == "decoration":
        generated = designer.scaffold_decoration(scaffold=inputs, **generate_params)
    elif mode.value == "superstructure":
        generated = designer.super_structure(
            core=inputs, attachment_point_depth=3, **generate_params
        )
    elif mode.value == "motif_extension":
        generated = designer.motif_extension(
            motif=inputs, min_length=len(inputs), **generate_params
        )
    elif mode.value == "morphing":
        generated = designer.scaffold_morphing(side_chains=inputs, **generate_params)
    elif mode.value == "linker":
        generated = designer.linker_generation(*inputs.split("."), **generate_params)

    data = {"smiles": generated}
    data = pd.DataFrame(generated)
    if inputs is not None:
        data["inputs"] = inputs
    data.to_csv(outfile, index=False)


@app.command()
def optim(
    checkpoint: Annotated[Optional[str], typer.Option()] = None,
    n_samples: Annotated[int, typer.Option(default=100)] = 100,
    n_trials: Annotated[int, typer.Option(default=1)] = 1,
    sanitize: Annotated[bool, typer.Option(default=True)] = True,
    allow_further_decomposition: Annotated[bool, typer.Option(default=False)] = False,
    seed: Annotated[int, typer.Option()] = 42,
    inputs: Annotated[str, typer.Option()] = None,
    objective: Annotated[
        DesignObjective, typer.Option(case_sensitive=False)
    ] = DesignObjective.denovo,
    batch_size: Annotated[int, typer.Option(default=32)] = 32,
    n_episodes: Annotated[int, typer.Option(default=100)] = 100,
    max_new_tokens: Annotated[int, typer.Option(default=100)] = 100,
    name: Annotated[str, typer.Option(default=...)] = None,
    outfile: Annotated[str, typer.Option(default=...)] = None,
):
    """Perform optimization under a given objective"""

    designer = SAFEDesign.load_default(verbose=False, model_dir=checkpoint)
    oracle = Oracle(name=objective.value)

    safe_tokenizer = designer.tokenizer
    tokenizer = safe_tokenizer.get_pretrained()
    model = AutoModelForCausalLMWithValueHead(designer.model)
    model.is_peft_model = False

    if name is None:
        name = f"safe-{objective.value}"

    ppo_config = {"batch_size": batch_size, "log_with": "wandb", "model_name": name}
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": max_new_tokens,
    }

    wandb.finish()
    trainer, trained_model = train(
        ppo_config,
        generation_kwargs,
        model,
        tokenizer,
        oracle,
        prefix=inputs,
        n_episodes=n_episodes,
        batch_size=batch_size,
    )

    designer.model = trained_model

    generate_params = {"n_samples_per_trial": n_samples, "n_trials": n_trials, "sanitize": sanitize}
    if inputs is not None:
        generated = designer.scaffold_decoration(
            scaffold=inputs,
            do_not_fragment_further=(not allow_further_decomposition),
            random_seed=seed,
            **generate_params,
        )
    else:
        generated = designer.de_novo_generation(**generate_params)

    data = {"smiles": generated}
    data = pd.DataFrame(generated)
    if inputs is not None:
        data["inputs"] = inputs
    data.to_csv(outfile, index=False)
