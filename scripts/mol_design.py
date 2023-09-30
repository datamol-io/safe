from typing import Optional
from typing_extensions import Annotated

import typer

import os
import torch
import safe as sf
import numpy as np
import pandas as pd
import datamol as dm
import itertools
import wandb
import fsspec

from enum import Enum
from loguru import logger
from tqdm.auto import tqdm
from safe import SAFEDesign
from tdc import Oracle
from tdc import Evaluator
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer()


class DesignMode(str, Enum):
    denovo = "denovo"
    superstructure = "superstructure"
    scaffold = "scaffold"
    motif = "motif"
    morphing = "morphing"
    linker = "linker"


class DesignObjective(str, Enum):
    clogp = "clogp"
    qed = "qed"
    sas = "sas"
    tpsa = "tpsa"
    mw = "mw"
    cns = "cns"


class TargetedReward:
    """Reward function for goal directed design

    reward = 1.0 / (1.0 + alpha * distance)
    """

    def __init__(self, objective="clogp", target=2, alpha=0.5):
        self.obj = objective
        self.target = target
        self.alpha = alpha
        self.cns_predictor = None
        if self.obj == "cns":
            import vdmpk.pka

            self.cns_predictor = vdmpk.pka.PkaPredictor.from_ada()

    @staticmethod
    def _fail_silently(fn, *arg, **kwargs):
        """Remains silent when the input function fails and return None instead"""
        try:
            return fn(*arg, **kwargs)
        except Exception:
            return None

    def __call__(self, smiles):
        if not isinstance(smiles, (list, tuple, np.ndarray)):
            smiles = [smiles]
        n_input = len(smiles)
        mols = [TargetedReward._fail_silently(dm.to_mol, x) for x in smiles]
        default_scores = np.zeros(len(mols))
        valid = np.array([v is not None for v in mols])
        if np.sum(valid) > 0:
            valid_mol = list(itertools.compress(mols, valid))
            out = self._metric(valid_mol)
            default_scores[valid] = out
        if n_input > 1:
            return default_scores.astype(float)
        return float(default_scores.flat[0])

    def _compute_cns(self, mols):
        out = []
        for mol in mols:
            try:
                results = self.cns_predictor.predict_pka(
                    mol=mol,
                    return_all_pka_values=False,
                    return_states=False,
                    return_mols=False,
                    return_clogd=False,
                    return_cns_score=True,
                    clogd_ph=7.4,
                    clogp=None,
                )
                out.append(results["mpo_score"])
            except:
                out.append(-1)
        return np.asarray(out)

    def _metric(self, mols):
        """Compute underlying metric objective for a set of smiles"""
        if self.obj == "cns":
            return self._compute_cns(mols)
        if self.obj == "clogp":
            out = [dm.descriptors.clogp(x) for x in mols]
        elif self.obj == "mw":
            out = [dm.descriptors.mw(x) for x in mols]
        elif self.obj == "sas":
            out = [dm.descriptors.sas(x) for x in mols]
        elif self.obj == "tpsa":
            out = [dm.descriptors.tpsa(x) for x in mols]
        elif self.obj == "qed":
            out = [dm.descriptors.qed(x) for x in mols]
        else:
            raise ValueError("Unknown objective")
        out = np.asarray(out)
        dist = np.abs(out - self.target)
        return 1.0 / (1.0 + self.alpha * dist)


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
    safe_encoder = sf.SAFEConverter()
    model_ref = create_reference_model(model)
    config = PPOConfig(**ppo_config)

    diversity_evaluator = Evaluator(name="Diversity")
    uniqueness_evaluator = Evaluator(name="Uniqueness")

    ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

    for _ in tqdm(range(n_episodes)):
        fragment = ""
        if isinstance(prefix, str):
            fragment = safe_encoder.encoder(
                prefix,
                canonical=False,
                randomize=True,
                constraints=None,
                allow_empty=True,
            )
            fragment = fragment.rstrip(".") + "."

        if isinstance(fragment, str):
            fragment = [fragment]

        batch_size = ppo_config.get("batch_size", 32)

        if len(fragment) < batch_size:
            fragment = np.random.choice(fragment, size=batch_size)

        game_data = {}
        game_data["query"] = fragment
        batch = tokenizer(
            [tokenizer.bos_token + x for x in fragment],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.pretrained_model.device)
        query_tensor = batch["input_ids"]
        response_tensor = ppo_trainer.generate(
            list(query_tensor), return_prompt=True, **generation_kwargs
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
        rewards = np.zeros(len(decoded_smiles))
        try:
            valid_position, valid_smiles = zip(
                *[(i, x) for i, x in enumerate(decoded_smiles) if x is not None]
            )
            batch_reward = oracle(list(valid_smiles))
            rewards[np.asarray(valid_position)] = batch_reward
        except Exception as e:
            logger.error(e)
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
    n_samples: Annotated[int, typer.Option()] = 1000,
    n_trials: Annotated[int, typer.Option()] = 1,
    sanitize: Annotated[bool, typer.Option()] = True,
    allow_further_decomposition: Annotated[bool, typer.Option()] = False,
    mode: Annotated[DesignMode, typer.Option(case_sensitive=False)] = DesignMode.denovo,
    inputs: Annotated[str, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 42,
    max_n: Annotated[int, typer.Option()] = -1,
    outfile: Annotated[str, typer.Option(default=...)] = None,
):
    """Sample molecule using SAFEDesign"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    designer = SAFEDesign.load_default(verbose=False, model_dir=checkpoint, device=device)
    generate_params = {
        "n_samples_per_trial": n_samples,
        "n_trials": n_trials,
        "sanitize": sanitize,
        "do_not_fragment_further": (not allow_further_decomposition),
        "random_seed": seed,
    }

    datas = []
    if mode.value == "denovo":
        generated = designer.de_novo_generation(
            n_samples_per_trial=n_samples, n_trials=n_trials, sanitize=sanitize
        )
        data = {"smiles": generated}
        data = pd.DataFrame(generated)
        data["mode"] = mode.value
        datas.append(data)

    else:
        inputs_df = pd.read_csv(inputs)
        input_list = inputs_df[mode.value].tolist()
        if max_n is not None and max_n > 0:
            input_list = input_list[:max_n]
        for cur_input in tqdm(input_list):
            try:
                if mode.value == "scaffold":
                    generated = designer.scaffold_decoration(scaffold=cur_input, **generate_params)
                elif mode.value == "superstructure":
                    generated = designer.super_structure(
                        core=cur_input, attachment_point_depth=3, **generate_params
                    )
                elif mode.value == "motif":
                    generated = designer.motif_extension(
                        motif=cur_input, min_length=len(inputs), **generate_params
                    )
                elif mode.value == "morphing":
                    generated = designer.scaffold_morphing(side_chains=cur_input, **generate_params)
                elif mode.value == "linker":
                    generated = designer.linker_generation(*cur_input.split("."), **generate_params)

                data = {"smiles": generated}
                data = pd.DataFrame(generated)
                if cur_input is not None:
                    data["inputs"] = cur_input
                data["mode"] = mode.value
                datas.append(data)
            except Exception as e:
                logger.exception(e)
    if len(datas) > 0:
        datas = pd.concat(datas, ignore_index=True)
        datas.to_csv(outfile, index=False)


@app.command()
def optim(
    checkpoint: Annotated[Optional[str], typer.Option()] = None,
    n_samples: Annotated[int, typer.Option()] = 500,
    n_trials: Annotated[int, typer.Option()] = 2,
    sanitize: Annotated[bool, typer.Option()] = True,
    allow_further_decomposition: Annotated[bool, typer.Option()] = True,
    seed: Annotated[int, typer.Option()] = 42,
    inputs: Annotated[str, typer.Option()] = None,
    task_id: Annotated[int, typer.Option()] = None,
    objective: Annotated[DesignObjective, typer.Option(case_sensitive=False)] = DesignObjective.mw,
    batch_size: Annotated[int, typer.Option()] = 100,
    n_episodes: Annotated[int, typer.Option()] = 100,
    target: Annotated[int, typer.Option()] = 350,
    alpha: Annotated[float, typer.Option()] = 0.5,
    learning_rate: Annotated[float, typer.Option()] = 5e-5,
    max_new_tokens: Annotated[int, typer.Option()] = 150,
    name: Annotated[str, typer.Option(default=...)] = None,
    outdir: Annotated[str, typer.Option(default=...)] = None,
):
    """Perform optimization under a given objective"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    designer = SAFEDesign.load_default(verbose=False, model_dir=checkpoint, device=device)

    safe_tokenizer = designer.tokenizer
    tokenizer = safe_tokenizer.get_pretrained()
    model = AutoModelForCausalLMWithValueHead(designer.model)
    model.is_peft_model = False
    TASKS = [
        ("mw", 350),
        ("mw", 400),
        ("mw", 450),
        ("clogp", 2),
        ("clogp", 4),
        ("clogp", 6),
        ("tpsa", 40),
        ("tpsa", 80),
        ("tpsa", 120),
        ("qed", 0.3),
        ("qed", 0.5),
        ("qed", 0.7),
        ("cns", None),
    ]
    if task_id is None:
        reward_fn = TargetedReward(objective=objective.value, target=target, alpha=alpha)
    else:
        reward_fn = TargetedReward(
            objective=TASKS[task_id][0], target=TASKS[task_id][1], alpha=alpha
        )
        name = name + f"-{TASKS[task_id][0]}-{TASKS[task_id][1]}"
        outdir = outdir.rstrip("/") + f"-{TASKS[task_id][0]}-{TASKS[task_id][1]}/"
    if inputs == "None":
        inputs = None
    if name is None:
        name = f"safe-{objective.value}"

    ppo_config = {
        "batch_size": batch_size,
        "log_with": "wandb",
        "model_name": "GPT",
        "tracker_project_name": name,
        "learning_rate": learning_rate,
    }

    generation_kwargs = {
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
        reward_fn,
        prefix=inputs or None,
        n_episodes=n_episodes,
        batch_size=batch_size,
    )

    trained_model.eval()
    designer.model = trained_model

    generate_params = {"n_samples_per_trial": n_samples, "n_trials": n_trials, "sanitize": sanitize}
    if inputs:
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

    with fsspec.open(os.path.join(outdir, f"data-{name}.csv"), "w", auto_mkdir=True) as IN:
        data.to_csv(IN, index=False)
    trained_model.save_pretrained(os.path.join(outdir, "model"))


if __name__ == "__main__":
    app()
