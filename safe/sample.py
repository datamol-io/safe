from typing import Union
from typing import List
from typing import Optional


from contextlib import suppress
from collections.abc import Mapping
from collections import Counter
from transformers.generation import PhrasalConstraint
from transformers import GenerationConfig
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.tokenizer import SAFETokenizer
from safe._constraint import LinkingLogitProcessor
from loguru import logger

import os
import random
import platformdirs
import datamol as dm
import safe as sf


class SAFEDesign:
    """Molecular generation using SAFE pretrained model"""

    _DEFAULT_MAX_LENGTH = 1024  # default max length used during training
    _DEFAULT_MODEL_PATH = platformdirs.user_cache_dir(
        "safe/default_model"
    )  # EN to change this as soon as possible

    def __init__(
        self,
        model: Union[SAFEDoubleHeadsModel, str],
        tokenizer: Union[str, SAFETokenizer],
        generation_config: Optional[Union[str, GenerationConfig]] = None,
        safe_encoder: Optional[sf.SAFEConverter] = None,
        verbose: bool = True,
    ):
        """SAFEDesign constructor

        Args:
            model: input SAFEDoubleHeadsModel to use for generation
            tokenizer: input SAFETokenizer to use for generation
            generation_config: input GenerationConfig to use for generation
            safe_encoder: custom safe encoder to use
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

        self.verbose = verbose
        self.safe_encoder = safe_encoder or sf.SAFEConverter()

    @classmethod
    def load_default(cls, verbose: bool = False) -> "SAFEDesign":
        """Load default SAFEGenerator model

        Args:
            verbose: whether to print out logging information during generation
        """
        model = SAFEDoubleHeadsModel.from_pretrained(cls._DEFAULT_MODEL_PATH)
        tokenizer = SAFETokenizer.load(os.path.join(cls._DEFAULT_MODEL_PATH, "tokenizer.json"))
        gen_config = GenerationConfig.from_pretrained(cls._DEFAULT_MODEL_PATH)
        return cls(model=model, tokenizer=tokenizer, generation_config=gen_config, verbose=verbose)

    def linker_generation(
        self,
        *groups: List[Union[str, dm.Mol]],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        sanitize: bool = False,
        do_not_fragment_further: Optional[bool] = False,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        """Perform linker generation using the pretrained SAFE model.
        Linker generation is really just scaffold morphing underlying.


        Args:
            groups: list of fragments to link together, they are joined in the order provided
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            kwargs: any argument to provide to the underlying generation function
        """
        if len(groups) < 2 and self.verbose:
            logger.warning(
                f"It's advised to have at most two groups when generating a linker, got {len(groups)} fragments"
            )
        side_chains = list(groups)
        return self.scaffold_morphing(
            side_chains=side_chains,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            sanitize=sanitize,
            do_not_fragment_further=do_not_fragment_further,
            random_seed=random_seed,
            **kwargs,
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
        side_chains = (
            [dm.to_mol(x) for x in side_chains]
            if isinstance(side_chains, list)
            else [dm.to_mol(side_chains)]
        )

        side_chains = ".".join([dm.to_smiles(x) for x in side_chains])
        if "*" not in side_chains and self.verbose:
            logger.warning(
                f"Side chain {side_chains} does not contain any dummy atoms, this might not be what you want"
            )

        rng = random.Random(random_seed)
        new_seed = rng.randint(1, 1000)

        total_sequences = []
        n_trials = n_trials or 1
        for _ in range(n_trials):
            with dm.without_rdkit_log():
                context_mng = (
                    sf.utils.attr_as(self.safe_encoder, "slicer", None)
                    if do_not_fragment_further
                    else suppress()
                )
                old_slicer = getattr(self.safe_encoder, "slicer", None)
                with context_mng:
                    try:
                        encoded_fragment = self.safe_encoder.encoder(
                            side_chains,
                            canonical=False,
                            randomize=True,
                            constraints=None,
                            allow_empty=True,
                            seed=new_seed,
                        )

                    except Exception as e:
                        if self.verbose:
                            logger.error(e)
                        raise sf.SAFEEncodeError(f"Failed to encode {side_chains}") from e
                    finally:
                        if old_slicer is not None:
                            self.safe_encoder.slicer = old_slicer

            prefix = encoded_fragment
            prefix, suffix = encoded_fragment.rsplit(".", 1)

            missing_closure = Counter(self.safe_encoder._find_branch_number(prefix))
            missing_closure = [f"{str(x)}" for x in missing_closure if missing_closure[x] % 2 == 1]
            word_ids = self.tokenizer.encode(missing_closure, add_special_tokens=False)
            word_ids = self.tokenizer.encode([suffix], add_special_tokens=False)
            constraints = [PhrasalConstraint(tkl) for tkl in word_ids]
            # if not kwargs.get("do_sample"):
            kwargs["constraints"] = constraints
            kwargs["num_beams"] = n_samples_per_trial
            kwargs["how"] = "beam"
            # kwargs["logits_processor"] = [LinkingLogitProcessor(prefix, self.tokenizer, max_link_tokens=None)]
            kwargs["early_stopping"] = False

            sequences = self._generate(n_samples=n_samples_per_trial, safe_prefix=prefix, **kwargs)
            sequences = self._decode_safe(sequences, canonical=True, remove_invalid=sanitize)
            total_sequences.extend(sequences)

        # then we should filter out molecules that do not match the requested
        if sanitize:
            total_sequences = sf.utils.filter_by_substructure_constraints(
                total_sequences, side_chains
            )
            if self.verbose:
                logger.info(
                    f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
                )
        return total_sequences

    def motif_extension(
        self,
        motif: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        sanitize: bool = False,
        do_not_fragment_further: Optional[bool] = False,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        """Perform motif extension using the pretrained SAFE model.
        Motif extension is really just scaffold decoration underlying.

        Args:
            scaffold: scaffold (with attachment points) to decorate
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules and check
            random_seed: random seed to use
            kwargs: any argument to provide to the underlying generation function
        """
        return self.scaffold_decoration(
            motif,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            sanitize=sanitize,
            do_not_fragment_further=do_not_fragment_further,
            random_seed=random_seed,
            add_dot=False,
            **kwargs,
        )

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
            old_verbose = self.verbose
            try:
                with sf.utils.attr_as(self, "verbose", False):
                    out = self._completion(
                        fragment=core,
                        n_samples_per_trial=n_samples_per_trial,
                        n_trials=1,
                        do_not_fragment_further=do_not_fragment_further,
                        sanitize=sanitize,
                        random_seed=random_seed,
                        **kwargs,
                    )
                    total_sequences.extend(out)
            except Exception as e:
                if old_verbose:
                    logger.error(e)

            finally:
                self.verbose = old_verbose

        if sanitize and self.verbose:
            logger.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
            )
        return total_sequences

    def scaffold_decoration(
        self,
        scaffold: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        do_not_fragment_further: Optional[bool] = False,
        sanitize: bool = False,
        random_seed: Optional[int] = None,
        add_dot=True,
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
            sanitize: whether to sanitize the generated molecules and check if the scaffold is still present
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
            add_dot=add_dot,
            **kwargs,
        )
        # if we require sanitization
        # then we should filter out molecules that do not match the requested
        if sanitize:
            total_sequences = sf.utils.filter_by_substructure_constraints(total_sequences, scaffold)
            if self.verbose:
                logger.info(
                    f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
                )
        return total_sequences

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

        def _decode_fn(x):
            return sf.decode(
                x,
                as_mol=False,
                fix=True,
                remove_added_hs=True,
                canonical=canonical,
                ignore_errors=True,
                remove_dummies=True,
            )

        if len(sequences) > 100:
            safe_strings = dm.parallelized(_decode_fn, sequences, n_jobs=-1)
        else:
            safe_strings = [_decode_fn(x) for x in sequences]
        if remove_invalid:
            safe_strings = [x for x in safe_strings if x is not None]

        return safe_strings

    def _completion(
        self,
        fragment: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        do_not_fragment_further: Optional[bool] = False,
        sanitize: bool = False,
        random_seed: Optional[int] = None,
        add_dot: Optional[bool] = False,
        is_safe: Optional[bool] = False,
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
            is_safe: whether the smiles is already encoded as a safe string
            add_dot: whether to add a dot at the end of the fragments to signal to the model that we want to generate a distinct fragment.
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
            if is_safe:
                encoded_fragment = fragment
            else:
                with dm.without_rdkit_log():
                    context_mng = (
                        sf.utils.attr_as(self.safe_encoder, "slicer", None)
                        if do_not_fragment_further
                        else suppress()
                    )
                    old_slicer = getattr(self.safe_encoder, "slicer", None)
                    with context_mng:
                        try:
                            encoded_fragment = self.safe_encoder.encoder(
                                fragment,
                                canonical=False,
                                randomize=True,
                                constraints=None,
                                allow_empty=True,
                                seed=new_seed,
                            )

                        except Exception as e:
                            if self.verbose:
                                logger.error(e)
                            raise sf.SAFEEncodeError(f"Failed to encode {fragment}") from e
                        finally:
                            if old_slicer is not None:
                                self.safe_encoder.slicer = old_slicer

            if add_dot and encoded_fragment.count("(") == encoded_fragment.count(")"):
                encoded_fragment = encoded_fragment.rstrip(".") + "."

            sequences = self._generate(
                n_samples=n_samples_per_trial, safe_prefix=encoded_fragment, **kwargs
            )

            sequences = self._decode_safe(sequences, canonical=True, remove_invalid=sanitize)
            total_sequences.extend(sequences)

        return total_sequences

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

        ??? note "Generation Parameters"
            From the hugging face documentation:

            * `greedy decoding` if how="greedy" and num_beams=1 and do_sample=False.
            * `multinomial sampling` if num_beams=1 and do_sample=True.
            * `beam-search decoding` if how="beam" and num_beams>1 and do_sample=False.
            * `beam-search multinomial` sampling by calling if beam=True, num_beams>1 and do_sample=True or how="random" and num_beams>1
            * `diverse beam-search decoding` if num_beams>1 and num_beam_groups>1

            It's also possible to ignore the 'how' shortcut and directly call the underlying generation methods using the proper arguments.
            Learn more here: https://huggingface.co/docs/transformers/v4.32.0/en/main_classes/text_generation#transformers.GenerationConfig
            Under the hood, the following will be applied depending on the arguments:

            * greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False
            * contrastive search by calling contrastive_search() if penalty_alpha>0. and top_k>1
            * multinomial sampling by calling sample() if num_beams=1 and do_sample=True
            * beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False
            * beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True
            * diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1
            * constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None
            * assisted decoding by calling assisted_decoding(), if assistant_model is passed to .generate()

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

        input_ids = safe_prefix
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

        elif how is not None and "beam" in how:
            num_beams = max((num_beams or 0), n_samples)

        is_greedy = how == "greedy" or (num_beams in [0, 1, None]) and do_sample is False

        kwargs["do_sample"] = do_sample
        if num_beams is not None:
            kwargs["num_beams"] = num_beams
        if num_beam_groups is not None:
            kwargs["num_beam_groups"] = num_beam_groups
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        kwargs["num_return_sequences"] = n_samples
        kwargs["max_length"] = max_length
        kwargs.setdefault("early_stopping", True)
        # EN we don't do anything with the score that the model might return on generate ...
        if not isinstance(input_ids, Mapping):
            input_ids = {"inputs": None}
        else:
            # EN: we remove the EOS token added before running the prediction
            # because the model output nonsense when we keep it.
            # I don't know why it works for text generation but not here
            for k in input_ids:
                input_ids[k] = input_ids[k][:, :-1]

        if is_greedy:
            kwargs["num_return_sequences"] = 1
            if num_beams is not None and num_beams > 1:
                raise ValueError("Cannot set num_beams|num_beam_groups > 1 for greedy")
            # under greedy decoding there can only be a single solution
            # we just duplicate the solution several time for efficiency
            outputs = self.model.generate(
                **input_ids,
                generation_config=self.generation_config,
                **kwargs,
            )
            sequences = [
                pretrained_tk.decode(outputs.sequences.squeeze(), skip_special_tokens=True)
            ] * n_samples

        else:
            outputs = self.model.generate(
                **input_ids,
                generation_config=self.generation_config,
                **kwargs,
            )
            sequences = pretrained_tk.batch_decode(outputs.sequences, skip_special_tokens=True)
        return sequences
