from typing import Union
from typing import List
from typing import Optional

import os
import random
import datamol as dm

import safe as sf

from transformers import GenerationConfig
from functools import partial
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.tokenizer import SAFETokenizer
from loguru import logger


class SAFEMolDesign:
    """Molecular generation using SAFE pretrained model"""

    _DEFAULT_MAX_LENGTH = 1024  # default max length used during training
    _DEFAULT_MODEL_PATH = "/home/emmanuel/safe/expts/model-v1"

    def __init__(
        self,
        model: Union[SAFEDoubleHeadsModel, str],
        tokenizer: Union[str, SAFETokenizer],
        generation_config: Optional[Union[str, GenerationConfig]] = None,
        safe_encoder: Optional[sf.SafeConverter] = None,
        verbose: bool = True,
    ):
        """SAFEMolDesign constructor

        Args:
            model: input SAFEDoubleHeadsModel to use for generation
            tokenizer: input SAFETokenizer to use for generation
            generation_config: input GenerationConfig to use for generation
            verbose: whether to print out logging information during generation
        """
        if isinstance(model, os.PathLike):
            model = SAFEDoubleHeadsModel.from_pretrained(model)

        if isinstance(tokenizer, os.PathLike):
            tokenizer = SAFETokenizer.load(tokenizer)

        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        if isinstance(generation_config, os.PathLike):
            generation_config = GenerationConfig.from_pretrained(generation_config)
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(model.config)
        self.generation_config = generation_config
        for special_token_id in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            if getattr(self.generation_config, special_token_id) is None:
                setattr(
                    self.generation_config, special_token_id, getattr(tokenizer, special_token_id)
                )

        self.safe_encoder = safe_encoder or sf.SafeConverter()

    @classmethod
    def load_default(cls) -> "SAFEMolDesign":
        """Load default SAFEGenerator model"""
        model = SAFEDoubleHeadsModel.from_pretrained(cls._DEFAULT_MODEL_PATH)
        tokenizer = SAFETokenizer.load(os.path.join(cls._DEFAULT_MODEL_PATH, "tokenizer.json"))
        gen_config = GenerationConfig.from_pretrained(cls._DEFAULT_MODEL_PATH)
        return cls(model=model, tokenizer=tokenizer, generation_config=gen_config)

    def linker_generation(self, *groups: List[Union[str, dm.Mol]], n_samples: int = 10):
        if len(groups) > 2 and self.verbose:
            logger.warning(
                f"It's advised to have at most two groups when generating a linker, got {len(groups)} fragments"
            )
        pass

    def motif_extension(self, motif: Union[str, dm.Mol], n_samples: int = 10):
        pass

    def super_structure(
        self,
        core: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        sanitize: bool = False,
        do_not_fragment_further: Optional[bool] = False,
        random_seed: Optional[int] = None,
        attachment_point_depth: Optional[int] = None,
        **kwargs,
    ):
        """Perform super structure generation using the pretrained SAFE model.

        To generate super-structure, we basically just create various attachment points to the input core,
        then perform scaffold decoration.

        Args:
            core: input substructure to use. We aim to generate super structures of this molecule
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of different attachment points to consider
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            attachment_point_depth: depth of opening the attachment points.
                Increasing this, means you increase the number of substitution point to consider.
            kwargs: any argument to provide to the underlying generation function
        """

        core = dm.to_mol(core)
        cores = sf.utils.list_individual_attach_points(core, depth=attachment_point_depth)
        # get the fully open mol, everytime too.
        cores.append(dm.to_smiles(dm.reactions.open_attach_points(core)))
        cores = list(set(cores))
        rng = random.Random(random_seed)
        rng.shuffle(cores)
        # now also get the single openining of an attachment point
        total_sequences = []
        n_trials = n_trials or 1
        for _ in range(n_trials):
            core = cores[_ % len(cores)]
            with sf.utils.attr_as(self, "verbose", False):
                out = self._completion(
                    fragment=core,
                    n_samples_per_trial=n_samples_per_trial,
                    n_trials=n_trials,
                    do_not_fragment_further=do_not_fragment_further,
                    sanitize=sanitize,
                    random_seed=random_seed,
                    **kwargs,
                )

                total_sequences.extend(out)

        if sanitize and self.verbose:
            logger.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
            )

    def scaffold_morphing(
        self,
        side_chains: Optional[Union[dm.Mol, str, List[Union[str, dm.Mol]]]] = None,
        mol: Optional[Union[dm.Mol, str]] = None,
        core: Optional[Union[dm.Mol, str]] = None,
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        sanitize: bool = False,
        do_not_fragment_further: Optional[bool] = False,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        """Perform scaffold morphing decoration using the pretrained SAFE model

        For scaffold morphing, we try to replace the core by a new one. If the side_chains are provided, we use them.
        If a combination of molecule and core is provided, then, we use them to extract the side chains and performing the
        scaffold morphing then.

        !!! note "Finding the side chains"
            The algorithm to find the side chains from core assumes that the core we get as input has attachment points.
            Those attachment points are never considered as part of the query, rather they are used to define the attachment points.
            See ~sf.utils.compute_side_chains for more information.

        Args:
            side_chains: side chains to use to perform scaffold morphing (joining as best as possible the set of fragments)
            mol: input molecules when side_chains are not provided
            core: core to morph into another scaffold
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            kwargs: any argument to provide to the underlying generation function
        """
        if side_chains is None:
            if mol is None and core is None:
                raise ValueError(
                    "Either side_chains OR mol+core should be provided for scaffold morphing"
                )
            side_chains = sf.trainer.utils.compute_side_chains(mol, core)
        side_chains = [dm.to_mol(x) for x in side_chains] if isinstance(side_chains, list) else [dm.to_mol(side_chains)]

        side_chains = ".".join([dm.to_smiles(x) for x in side_chains])
        if "*" not in side_chains and self.verbose:
            logger.warning(
                f"Side chain {side_chains} does not contain any dummy atoms, this might not be what you want"
            )

        return self._completion(
            fragment=side_chains,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            do_not_fragment_further=do_not_fragment_further,
            sanitize=sanitize,
            random_seed=random_seed,
            **kwargs,
        )

        # check for scaffold morphing validity

    def scaffold_decoration(
        self,
        scaffold: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        do_not_fragment_further: Optional[bool] = False,
        sanitize: bool = False,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        """Perform scaffold decoration using the pretrained SAFE model

        For scaffold decoration, we basically starts with a prefix with the attachment point.
        We first convert the prefix into valid safe string.

        Args:
            scaffold: scaffold (with attachment points) to decorate
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            kwargs: any argument to provide to the underlying generation function
        """

        total_sequences = self._completion(
            fragment=scaffold,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            do_not_fragment_further=do_not_fragment_further,
            sanitize=sanitize,
            random_seed=random_seed,
            **kwargs,
        )
        # if we require sanitization
        # then we should filter out molecules that do not match the requested
        if sanitize:
            total_sequences = sf.utils.filter_by_substructure_constraints(total_sequences, scaffold)
        if sanitize and self.verbose:
            logger.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
            )

    def de_novo_generation(self, n_samples_per_trial: int = 10, sanitize: bool = False, **kwargs):
        """Perform de novo generation using the pretrained SAFE model.

        De novo generation is equivalent to not having any prefix.

        Args:
            n_samples_per_trial: number of new molecules to generate
            sanitize: whether to perform sanitization, aka, perform control to ensure what is asked is what is returned
            kwargs: any argument to provide to the underlying generation function
        """
        # EN: lazy programming much ?
        kwargs.setdefault("how", "random")
        if kwargs["how"] != "random" and not kwargs.get("do_sample"):
            logger.warning(
                "I don't think you know what you are doing ... for de novo generation `do_sample=True` or `how='random'` is expected !"
            )

        sequences = self._generate(n_samples=n_samples_per_trial, **kwargs)
        total_sequences = self._decode_safe(sequences, canonical=True, remove_invalid=sanitize)
        if sanitize and self.verbose:
            logger.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial} ({len(total_sequences)*100/n_samples_per_trial:.2f} %)  generated molecules are valid !"
            )
        return total_sequences

    def _decode_safe(
        self, sequences: List[str], canonical: bool = True, remove_invalid: bool = False
    ):
        """Decode a safe sequence into a molecule
        Args:
            sequence: safe sequence to decode
            canonical: whether to return canonical sequence
            remove_invalid: whether to remove invalid safe strings or keep them
        """

        decode_fn = partial(
            sf.decode(
                as_mol=False,
                fix=True,
                remove_added_hs=True,
                canonical=canonical,
                ignore_errors=True,
                remove_dummies=True,
            )
        )
        safe_strings = dm.parallelized(decode_fn, sequences)
        if remove_invalid:
            safe_strings = [x for x in safe_strings if x is not None]
        return safe_strings

    def _generate(
        self,
        n_samples: int = 1,
        safe_prefix: Optional[str] = None,
        max_length: Optional[int] = 100,
        how: Optional[str] = "random",
        num_beams: Optional[int] = None,
        num_beam_groups: Optional[int] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ):
        """Sample a new sequence using the underlying hugging face model.
        This emulates the izanagi sampling models, if you wish to retain the hugging face generation
        behaviour, either call the hugging face functions directly or overwrite this function

        !!! note
            From the hugging face documentation:

            * `greedy decoding` if how="greedy" and num_beams=1 and do_sample=False.
            * `multinomial sampling` if num_beams=1 and do_sample=True.
            * `beam-search decoding` if how="beam" and num_beams>1 and do_sample=False.
            * `beam-search multinomial` sampling by calling if beam=True, num_beams>1 and do_sample=True or how="random" and num_beams>1
            * `diverse beam-search decoding` if num_beams>1 and num_beam_groups>1.

        Args:
            n_samples: number of sequences to return
            safe_prefix: Prefix to use in sampling, should correspond to a safe fragment
            max_length : maximum length of sampled sequence
            how: which sampling method to use: "beam", "greedy" or "random". Can be used to control other parameters by setting defaults
            num_beams: number of beams for beam search. 1 means no beam search, unless beam is specified then max(n_samples, num_beams) is used
            num_beam_groups: number of beam groups for diverse beam search
            do_sample: whether to perform random sampling or not, equivalent to setting random to True
            kwargs: any additional keyword argument to pass to the underlying sampling `generate`  from hugging face transformer

        Returns:
            samples: list of sampled molecules, including failed validation

        """
        pretrained_tk = self.tokenizer.get_pretrained()
        if getattr(pretrained_tk, "model_max_length") is None:
            setattr(
                pretrained_tk,
                "model_max_length",
                self._DEFAULT_MAX_LENGTH,  # this was the defaul
            )

        if isinstance(safe_prefix, str):
            # EN: should we address the special token issues
            input_ids = pretrained_tk(
                safe_prefix,
                return_tensors="pt",
            )

        num_beams = num_beams or None
        do_sample = do_sample or False
        if how == "random":
            do_sample = True
        elif "beam" in how:
            num_beams = max((num_beams or 0), n_samples)

        is_greedy = how == "greedy" or (num_beams in [0, 1, None]) and do_sample is False

        kwargs["generation_config"] = self.generation_config
        kwargs["do_sample"] = do_sample
        kwargs["num_beams"] = num_beams
        kwargs["num_beam_groups"] = num_beam_groups
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        kwargs["num_return_sequences"] = n_samples
        kwargs["max_length"] = max_length
        kwargs.setdefault("early_stopping", True)

        # EN we don't do anything with the score that the model might return on generate ...
        if is_greedy:
            kwargs["num_return_sequences"] = 1
            if num_beams is not None and num_beams > 1:
                raise ValueError("Cannot set num_beams|num_beam_groups > 1 for greedy")
            # under greedy decoding there can only be a single solution
            # we just duplicate the solution several time for efficiency
            outputs = self.model.generate(
                input_ids,
                **kwargs,
            )
            sequences = [
                pretrained_tk.decode(outputs.sequences.squeeze(), skip_special_tokens=True)
            ] * n_samples

        else:
            outputs = self.model.generate(
                input_ids,
                **kwargs,
            )
            sequences = pretrained_tk.batch_decode(outputs.sequences, skip_special_tokens=True)
        return sequences

    def _completion(
        self,
        fragment: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        do_not_fragment_further: Optional[bool] = False,
        sanitize: bool = False,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        """Perform sentence completion using a prefix fragment

        Args:
            scaffold: scaffold (with attachment points) to decorate
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            kwargs: any argument to provide to the underlying generation function
        """

        # EN: lazy programming much ?
        kwargs.setdefault("how", "random")
        if kwargs["how"] != "random" and not kwargs.get("do_sample"):
            logger.warning(
                "I don't think you know what you are doing ... for de novo generation `do_sample=True` or `how='random'` is expected !"
            )

        # Step 1: we conver the fragment into the relevant safe string format
        # we use the provided safe encoder with the slicer that was expected

        rng = random.Random(random_seed)
        new_seed = rng.randint(1, 1000)

        total_sequences = []
        n_trials = n_trials or 1
        for _ in range(n_trials):
            with dm.without_rdkit_log():
                try:
                    if not do_not_fragment_further:
                        with sf.utils.attr_as(self.safe_encoder, "slicer", None):
                            encoded_fragment = self.safe_encoder.encoder(
                                fragment,
                                canonical=False,
                                randomize=True,
                                constraints=None,
                                seed=new_seed,
                            )
                    else:
                        encoded_fragment = self.safe_encoder.encoder(
                            fragment,
                            canonical=False,
                            randomize=True,
                            constraints=None,
                            seed=new_seed,
                        )
                except Exception as e:
                    raise sf.SafeEncodeError(f"Failed to encode {fragment}") from e

            # we add a bit of randomization, just for fun

            sequences = self._generate(
                n_samples=n_samples_per_trial, safe_prefix=encoded_fragment, **kwargs
            )
            sequences = self._decode_safe(sequences, canonical=True, remove_invalid=sanitize)
            total_sequences.extend(sequences)

        return total_sequences
