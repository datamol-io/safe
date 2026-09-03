import copy
import itertools
import os
import random
import re
import warnings
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from typing import List, Optional, Union, Any, Dict

import datamol as dm
import torch
from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

import safe as sf
from safe._connect import analyze
from safe.tokenizer import SAFETokenizer
from safe.trainer.model import SAFEDoubleHeadsModel
from safe._pattern import PatternConstraint, PatternSampler


class ScaffoldConnectivityLogitsProcessor(LogitsProcessor):
    """Steer scaffold completion toward a single connected molecule.

    During completion tasks a plain language model tends to close the scaffold's
    attachment points and then keep emitting extra ``.``-fragments that never
    bond back, yielding the scaffold plus spurious disconnected pieces. This
    processor tracks ring-closure connectivity (via :func:`safe._connect.analyze`)
    over the tokens generated so far and, at each step:

    * forces the end-of-sequence token once the sequence is a single connected
      component with balanced parentheses and no open ring labels;
    * suppresses the end-of-sequence token while the molecule is still
      incomplete; and
    * forbids starting a new ``.``-fragment while the fragment being written has
      not yet attached to the scaffold, so free fragments cannot pile up.

    Works with multinomial, greedy and beam search, and needs no retraining. It
    is stateless (recomputed from ``input_ids`` each step), so it is safe to use
    batched and inside a reinforcement-learning sampling loop.
    """

    def __init__(self, tokenizer, prompt_len: int):
        self._id2tok = {idx: tok for tok, idx in tokenizer.get_vocab().items()}
        self._specials = frozenset(tokenizer.all_special_tokens)
        self._prompt_len = prompt_len
        self._eos = tokenizer.eos_token_id
        self._dot_id = tokenizer.convert_tokens_to_ids(".")
        unk = tokenizer.unk_token_id
        self._has_dot = self._dot_id is not None and self._dot_id != unk

    def _tokens(self, row) -> List[str]:
        return [self._id2tok.get(int(i), "") for i in row]

    def __call__(self, input_ids, scores):
        neg_inf = torch.finfo(scores.dtype).min
        for b in range(input_ids.shape[0]):
            tokens = self._tokens(input_ids[b].tolist())
            decision = analyze(
                tokens[: self._prompt_len],
                tokens[self._prompt_len :],
                special_tokens=self._specials,
            )
            if decision.complete:
                # A valid connected molecule is reached: force EOS so the model
                # cannot append spurious disconnected fragments.
                forced = scores[b, self._eos].clone()
                scores[b, :] = neg_inf
                scores[b, self._eos] = forced
                continue
            # Incomplete: never stop here, and do not open a new fragment before
            # the current one has attached to the scaffold.
            if self._eos is not None:
                scores[b, self._eos] = neg_inf
            if self._has_dot and not decision.current_attached:
                scores[b, self._dot_id] = neg_inf
        return scores


class SAFEDesign:
    """Design molecules with a pretrained SAFE language model."""

    _DEFAULT_MAX_LENGTH = 1024  # default max length used during training
    _DEFAULT_MODEL_PATH = "datamol-io/safe-gpt"
    _DEFAULT_MODEL_REVISION = "3d5fa0988383e898d5ac5db7cd52bf715bc37061"
    _REFINE_OVERSAMPLE_FACTOR = 3
    _CONSTRAINED_GENERATION_BACKEND = (
        "transformers-community/constrained-beam-search",
        "07b2f120b7db38f1d7bac617ad65ea130508f297",
        "SAFE_CONSTRAINED_GENERATION_BACKEND",
    )

    def __init__(
        self,
        model: Union[SAFEDoubleHeadsModel, str],
        tokenizer: Union[str, SAFETokenizer],
        generation_config: Optional[Union[str, GenerationConfig]] = None,
        safe_encoder: Optional[sf.SAFEConverter] = None,
        verbose: bool = True,
    ):
        """SAFEDesign constructor

        !!! info
            Design methods in SAFE are not deterministic when it comes to the token sampling step.
            If a method accepts a `random_seed`, it's for the SAFE-related algorithms and not the
            sampling from the autoregressive model. To ensure you get a deterministic sampling,
            please set the seed at the `transformers` package level.

            ```python
            import safe as sf
            import transformers
            my_seed = 100
            designer = sf.SAFEDesign(...)

            transformers.set_seed(100) # use this before calling a design function
            designer.linker_generation(...)
            ```


        Args:
            model: input SAFEDoubleHeadsModel to use for generation
            tokenizer: input SAFETokenizer to use for generation
            generation_config: input GenerationConfig to use for generation
            safe_encoder: custom safe encoder to use
            verbose: whether to print out logging information during generation
        """

        if isinstance(model, (str, os.PathLike)):
            model = SAFEDoubleHeadsModel.from_pretrained(model)

        if isinstance(tokenizer, (str, os.PathLike)):
            tokenizer = SAFETokenizer.load(tokenizer)

        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        if isinstance(generation_config, (str, os.PathLike)):
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
        self._constrained_generator = None

    @classmethod
    def _candidate_count(cls, requested: int, refine: bool) -> int:
        """Return the number of raw candidates to sample for a design trial."""
        return requested * cls._REFINE_OVERSAMPLE_FACTOR if refine else requested

    @staticmethod
    def _finalize_samples(sequences: List[Optional[str]], limit: int, refine: bool):
        """Deduplicate valid samples in generation order when quality mode is enabled."""
        if not refine:
            return sequences
        unique = list(dict.fromkeys(sequence for sequence in sequences if sequence is not None))
        finalized = unique[:limit]
        if len(finalized) < limit:
            # refine cannot guarantee the requested count; surface the
            # shortfall instead of silently returning fewer candidates.
            logger.warning(
                f"refine produced only {len(finalized)} unique valid candidate(s) "
                f"out of the {limit} requested; consider increasing n_samples_per_trial "
                f"or n_trials."
            )
        return finalized

    def _load_constrained_generation_backend(self):
        """Load the pinned backend required by model-only linker generation."""
        if getattr(self, "_constrained_generator", None) is not None:
            return self._constrained_generator

        repo_id, revision, env_name = self._CONSTRAINED_GENERATION_BACKEND
        backend = os.getenv(env_name, repo_id)
        load_kwargs = {"trust_remote_code": True}
        if not os.path.isdir(backend):
            load_kwargs["revision"] = revision
        self._constrained_generator = self.model.load_custom_generate(backend, **load_kwargs)
        return self._constrained_generator

    @classmethod
    def load_from_wandb(
        cls, artifact_path: str, device: Optional[str] = None, verbose: bool = True, **kwargs: Any
    ) -> "SAFEDesign":
        """Load a SAFE model and tokenizer from a Weights & Biases artifact.

        When ``SAFE_MODEL_ROOT`` is set, the artifact is downloaded into that
        directory.

        Args:
            artifact_path: The path to the wandb artifact in the format `entity/project/artifact:version`.
            device: The device where the model should be loaded ('cpu' or 'cuda'). If None, it defaults to the available device.
            verbose: Whether to print out logging information during generation.

        Returns:
            SAFEDesign: An instance of SAFEDesign class with the model, tokenizer, and generation config loaded from wandb.
        """
        import wandb

        artifact_path = artifact_path.replace("wandb://", "")

        # Parse the artifact path to extract project and artifact name
        parts = artifact_path.split("/", 1)
        if len(parts) > 1:
            project_name, artifact_name = parts
        else:
            project_name = os.getenv("SAFE_WANDB_PROJECT", "safe-models")
            artifact_name = artifact_path

        if ":" not in artifact_name:
            artifact_name += ":latest"

        artifact_path = f"{project_name}/{artifact_name}"

        # Check if SAFE_MODEL_ROOT environment variable is defined
        cache_path = os.getenv("SAFE_MODEL_ROOT", None)
        if cache_path is not None:
            # Ensure the cache path exists
            cache_path = Path(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            artifact_subfolder = artifact_path.replace("/", "_").replace(":", "_")
            cache_dir = cache_path / artifact_subfolder
            cache_path = cache_dir.as_posix()

        api = wandb.Api()
        # Download the artifact from wandb to the cache directory
        artifact = api.artifact(artifact_path, type="model")
        artifact_dir = artifact.download(root=cache_path)

        # Load the model, tokenizer, and generation config from the artifact directory
        model = SAFEDoubleHeadsModel.from_pretrained(artifact_dir)
        tokenizer = SAFETokenizer.from_pretrained(artifact_dir)
        gen_config = GenerationConfig.from_pretrained(artifact_dir)

        # Move model to the specified device if provided
        if device is not None:
            model = model.to(device)

        return cls(
            model=model,
            tokenizer=tokenizer,
            generation_config=gen_config,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    def load_default(
        cls,
        model_dir: Optional[str] = None,
        model_revision: Optional[str] = None,
        device: str = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "SAFEDesign":
        """Load default SAFEGenerator model

        Args:
            verbose: whether to print out logging information during generation
            model_dir: Optional path to model folder to use instead of the default one.
                If provided the tokenizer should be in the model_dir named as `tokenizer.json`
            model_revision: Hugging Face revision to load. The reviewed default
                model revision is pinned when `model_dir` is omitted.
            device: optional device where to move the model
            kwargs: any additional argument to pass to the init function
        """
        use_default_model = model_dir is None or not model_dir
        if use_default_model:
            model_dir = cls._DEFAULT_MODEL_PATH
            model_revision = model_revision or cls._DEFAULT_MODEL_REVISION
        load_kwargs = {"revision": model_revision} if model_revision is not None else {}
        model = SAFEDoubleHeadsModel.from_pretrained(model_dir, **load_kwargs)
        tokenizer = SAFETokenizer.from_pretrained(model_dir, **load_kwargs)
        gen_config = GenerationConfig.from_pretrained(model_dir, **load_kwargs)
        if device is not None:
            model = model.to(device)
        return cls(
            model=model,
            tokenizer=tokenizer,
            generation_config=gen_config,
            verbose=verbose,
            **kwargs,
        )

    def linker_generation(
        self,
        *groups: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        sanitize: bool = False,
        do_not_fragment_further: Optional[bool] = True,
        random_seed: Optional[int] = None,
        model_only: Optional[bool] = False,
        refine: bool = False,
        **kwargs: Optional[Dict[Any, Any]],
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
            model_only: whether to use the model only ability and nothing more.
            refine: quality mode. Oversample, then
                return only valid, deduplicated molecules up to the requested count; for
                completion tasks the result is also constrained to a single connected molecule.
            kwargs: any argument to provide to the underlying generation function
        """
        side_chains = list(groups)

        if len(side_chains) != 2:
            raise ValueError(
                "Linker generation only works when providing two groups as side chains"
            )

        return self._fragment_linking(
            side_chains=side_chains,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            sanitize=sanitize,
            do_not_fragment_further=do_not_fragment_further,
            random_seed=random_seed,
            is_linking=True,
            model_only=model_only,
            refine=refine,
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
        do_not_fragment_further: Optional[bool] = True,
        random_seed: Optional[int] = None,
        refine: bool = False,
        **kwargs: Optional[Dict[Any, Any]],
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
            refine: quality mode. Oversample, then
                return only valid, deduplicated molecules up to the requested count; for
                completion tasks the result is also constrained to a single connected molecule.
            kwargs: any argument to provide to the underlying generation function
        """

        return self._fragment_linking(
            side_chains=side_chains,
            mol=mol,
            core=core,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            sanitize=sanitize,
            do_not_fragment_further=do_not_fragment_further,
            random_seed=random_seed,
            is_linking=False,
            refine=refine,
            **kwargs,
        )

    def _fragment_linking(
        self,
        side_chains: Optional[Union[dm.Mol, str, List[Union[str, dm.Mol]]]] = None,
        mol: Optional[Union[dm.Mol, str]] = None,
        core: Optional[Union[dm.Mol, str]] = None,
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        sanitize: bool = False,
        do_not_fragment_further: Optional[bool] = False,
        random_seed: Optional[int] = None,
        is_linking: Optional[bool] = False,
        model_only: Optional[bool] = False,
        refine: bool = False,
        **kwargs: Optional[Dict[Any, Any]],
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
            is_linking: whether it's a linking task or not.
                For linking tasks, we use a different custom strategy of completing up to the attachment signal
            model_only: whether to use the model only ability and nothing more. Only relevant when doing linker generation
            refine: quality mode. Oversample, then
                return only valid, deduplicated molecules up to the requested count; for
                completion tasks the result is also constrained to a single connected molecule.
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

        total_sequences = []
        n_trials = n_trials or 1
        candidates_per_trial = self._candidate_count(n_samples_per_trial, refine)
        validate = sanitize or refine
        for _ in tqdm(range(n_trials), disable=(not self.verbose), leave=False):
            new_seed = rng.randint(1, 2**32 - 1)
            with dm.without_rdkit_log():
                context_mng = (
                    sf.utils.attr_as(self.safe_encoder, "slicer", None)
                    if do_not_fragment_further
                    else suppress()
                )
                with context_mng:
                    try:
                        encoded_fragment = self.safe_encoder.encoder(
                            side_chains,
                            canonical=False,
                            randomize=False,
                            constraints=None,
                            allow_empty=True,
                            seed=new_seed,
                        )

                    except Exception as e:
                        if self.verbose:
                            logger.error(e)
                        raise sf.SAFEEncodeError(f"Failed to encode {side_chains}") from e

            fragments = encoded_fragment.split(".")
            branch_positions = self.safe_encoder._find_branch_number_positions(encoded_fragment)
            branch_counts = Counter(label for label, _ in branch_positions)
            missing_closure_labels = [
                label for label, count in branch_counts.items() if count % 2 == 1
            ]
            missing_closure = [
                self.safe_encoder._format_ring_closure(label) for label in missing_closure_labels
            ]
            closure_pos = [
                position for label, position in branch_positions if label in missing_closure_labels
            ]
            fragment_pos = [m.start() for m in re.finditer(r"\.", encoded_fragment)]
            if not closure_pos or not fragment_pos:
                raise ValueError(
                    "Side chains must contain terminal attachment points "
                    "distributed across at least two fragments"
                )
            min_pos = 0
            while min_pos < len(fragment_pos) and fragment_pos[min_pos] < closure_pos[0]:
                min_pos += 1
            min_pos += 1
            max_pos = len(fragment_pos)
            while max_pos > 0 and fragment_pos[max_pos - 1] > closure_pos[-1]:
                max_pos -= 1

            if min_pos > max_pos:
                raise ValueError(
                    "Attachment points must be distributed across at least two fragments"
                )

            split_index = rng.randint(min_pos, max_pos)
            prefix, suffixes = ".".join(fragments[:split_index]), ".".join(fragments[split_index:])

            prefix_branch_counts = Counter(self.safe_encoder._find_branch_number(prefix))
            suffix_branch_counts = Counter(self.safe_encoder._find_branch_number(suffixes))
            missing_prefix_closure = (
                ["."]
                + [
                    token
                    for label, token in zip(missing_closure_labels, missing_closure)
                    if label not in prefix_branch_counts
                ]
                + ["."]
            )
            missing_suffix_closure = (
                ["."]
                + [
                    token
                    for label, token in zip(missing_closure_labels, missing_closure)
                    if label not in suffix_branch_counts
                ]
                + ["."]
            )

            # Each permutation is one complete alternative phrase. The old
            # batch encoding produced a list of one-token alternatives and
            # made a dot sufficient to satisfy the constraint, so the required
            # ring closure could silently be absent.
            constraint_alternatives = [
                self.tokenizer.encode("".join(permutation), add_special_tokens=False)
                for permutation in sorted(set(itertools.permutations(missing_closure + ["."])))
            ]

            prefix_kwargs = kwargs.copy()
            suffix_kwargs = prefix_kwargs.copy()

            if is_linking and model_only:
                if len(missing_prefix_closure) < 3 or len(missing_suffix_closure) < 3:
                    raise ValueError(
                        "Each linker side must expose at least one unmatched attachment label"
                    )
                for _kwargs in [prefix_kwargs, suffix_kwargs]:
                    _kwargs.setdefault("how", "beam")
                    _kwargs.setdefault("num_beams", candidates_per_trial)
                    _kwargs.setdefault("do_sample", False)

                # We first generate a fragment that contains one complete
                # closure phrase. ``force_words_ids`` is the stable serialized
                # representation understood by the pinned constrained-beam
                # backend; it also avoids importing constraint classes removed
                # from Transformers 5.
                prefix_kwargs["force_words_ids"] = [constraint_alternatives]
                suffix_kwargs["force_words_ids"] = [constraint_alternatives]

                prefix_sequences = self._generate(
                    n_samples=candidates_per_trial, safe_prefix=prefix, **prefix_kwargs
                )
                suffix_sequences = self._generate(
                    n_samples=candidates_per_trial, safe_prefix=suffixes, **suffix_kwargs
                )

                prefix_sequences = [
                    self._find_fragment_cut(x, prefix, missing_prefix_closure[1])
                    for x in prefix_sequences
                ]
                suffix_sequences = [
                    self._find_fragment_cut(x, suffixes, missing_suffix_closure[1])
                    for x in suffix_sequences
                ]

                linkers = sorted(
                    set(x for x in prefix_sequences + suffix_sequences if x),
                    reverse=True,
                )
                sequences = [f"{prefix}.{linker}.{suffixes}" for linker in linkers]
                # Public design methods return decoded SMILES. The previous
                # ``+=`` leaked the intermediate SAFE strings alongside their
                # decoded molecules and doubled the apparent sample count.
                sequences = self._decode_safe(sequences, canonical=True, remove_invalid=validate)

            else:
                mol_linker_slicer = sf.utils.MolSlicer(
                    shortest_linker=(not is_linking), require_ring_system=(not is_linking)
                )
                prefix_smiles = sf.decode(prefix, remove_dummies=False, as_mol=False)
                suffix_smiles = sf.decode(suffixes, remove_dummies=False, as_mol=False)

                prefix_sequences = self._generate(
                    n_samples=candidates_per_trial, safe_prefix=prefix + ".", **prefix_kwargs
                )
                suffix_sequences = self._generate(
                    n_samples=candidates_per_trial, safe_prefix=suffixes + ".", **suffix_kwargs
                )

                prefix_sequences = self._decode_safe(
                    prefix_sequences, canonical=True, remove_invalid=True
                )
                suffix_sequences = self._decode_safe(
                    suffix_sequences, canonical=True, remove_invalid=True
                )
                sequences = self.__mix_sequences(
                    prefix_sequences,
                    suffix_sequences,
                    prefix_smiles,
                    suffix_smiles,
                    candidates_per_trial,
                    mol_linker_slicer,
                )

            total_sequences.extend(sequences)

        # then we should filter out molecules that do not match the requested
        if validate:
            total_sequences = sf.utils.filter_by_substructure_constraints(
                total_sequences, side_chains
            )
            if self.verbose:
                logger.info(
                    f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
                )
        return self._finalize_samples(
            total_sequences,
            n_samples_per_trial * n_trials,
            refine,
        )

    def motif_extension(
        self,
        motif: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        sanitize: bool = False,
        do_not_fragment_further: Optional[bool] = True,
        random_seed: Optional[int] = None,
        refine: bool = False,
        **kwargs: Optional[Dict[Any, Any]],
    ):
        """Deprecated alias for :meth:`scaffold_decoration`."""
        warnings.warn(
            "motif_extension() is an alias of scaffold_decoration() and will be removed in "
            "SAFE 2.0; call scaffold_decoration() directly.",
            FutureWarning,
            stacklevel=2,
        )
        return self.scaffold_decoration(
            motif,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            sanitize=sanitize,
            do_not_fragment_further=do_not_fragment_further,
            random_seed=random_seed,
            add_dot=True,
            refine=refine,
            **kwargs,
        )

    def super_structure(
        self,
        core: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        sanitize: bool = False,
        do_not_fragment_further: Optional[bool] = True,
        random_seed: Optional[int] = None,
        attachment_point_depth: Optional[int] = None,
        refine: bool = False,
        **kwargs: Optional[Dict[Any, Any]],
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
            refine: quality mode. Oversample, then
                return only valid, deduplicated molecules up to the requested count; for
                completion tasks the result is also constrained to a single connected molecule.
            kwargs: any argument to provide to the underlying generation function
        """

        core = dm.to_mol(core)
        # Keep the original core: ``core`` is reassigned inside the trial loop
        # below, but the requested substructure to enforce is this input.
        requested_core = core
        cores = sf.utils.list_individual_attach_points(core, depth=attachment_point_depth)
        # get the fully open mol, everytime too.
        cores.append(dm.to_smiles(dm.reactions.open_attach_points(core)))
        cores = list(dict.fromkeys(cores))
        rng = random.Random(random_seed)
        rng.shuffle(cores)
        # now also get the single openining of an attachment point
        total_sequences = []
        n_trials = n_trials or 1
        for _ in tqdm(range(n_trials), disable=(not self.verbose), leave=False):
            core = cores[_ % len(cores)]
            try:
                with sf.utils.attr_as(self, "verbose", False):
                    out = self._completion(
                        fragment=core,
                        n_samples_per_trial=n_samples_per_trial,
                        n_trials=1,
                        do_not_fragment_further=do_not_fragment_further,
                        sanitize=sanitize or refine,
                        random_seed=rng.randint(1, 2**32 - 1),
                        refine=refine,
                        **kwargs,
                    )
                    total_sequences.extend(out)
            except (sf.SAFEEncodeError, ValueError) as e:
                if self.verbose:
                    logger.error(e)

        # Match the other constrained methods: verify the requested core is
        # actually present in the generated superstructures.
        if sanitize or refine:
            total_sequences = sf.utils.filter_by_substructure_constraints(
                total_sequences, requested_core
            )
        if sanitize and self.verbose:
            logger.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
            )
        return self._finalize_samples(
            total_sequences,
            n_samples_per_trial * n_trials,
            refine,
        )

    def scaffold_decoration(
        self,
        scaffold: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: Optional[int] = 1,
        do_not_fragment_further: Optional[bool] = True,
        sanitize: bool = False,
        random_seed: Optional[int] = None,
        add_dot: Optional[bool] = True,
        refine: bool = False,
        **kwargs: Optional[Dict[Any, Any]],
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
            refine: quality mode. Oversample, then
                return only valid, deduplicated molecules up to the requested count; for
                completion tasks the result is also constrained to a single connected molecule.
            kwargs: any argument to provide to the underlying generation function
        """

        n_trials = n_trials or 1
        total_sequences = self._completion(
            fragment=scaffold,
            n_samples_per_trial=n_samples_per_trial,
            n_trials=n_trials,
            do_not_fragment_further=do_not_fragment_further,
            sanitize=sanitize or refine,
            random_seed=random_seed,
            add_dot=add_dot,
            refine=refine,
            **kwargs,
        )
        # if we require sanitization
        # then we should filter out molecules that do not match the requested
        if sanitize or refine:
            total_sequences = sf.utils.filter_by_substructure_constraints(total_sequences, scaffold)
            if self.verbose:
                logger.info(
                    f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %)  generated molecules are valid !"
                )
        return self._finalize_samples(
            total_sequences,
            n_samples_per_trial * n_trials,
            refine,
        )

    def pattern_decoration(
        self,
        scaffold: Union[str, dm.Mol],
        n_samples_per_trial: int = 10,
        n_trials: int = 1,
        do_not_fragment_further: bool = True,
        sanitize: bool = False,
        random_seed: Optional[int] = None,
        add_dot: bool = True,
        n_scaff_random: Optional[int] = 3,
        n_scaff_samples: Optional[int] = 10,
        scaff_temperature: float = 1.0,
        refine: bool = False,
        **kwargs: Optional[Dict[Any, Any]],
    ) -> List[str]:
        """
        Perform pattern decoration using the pretrained SAFE model. The pattern decoration algorithm works by first examplifying the patterns
        as a set of scaffold then performing scaffold decoration on each scaffold.

        !!! warning
            Designing molecules from a given molecule pattern is more challenging than fragment-constrained design.
            SAFE does not currently support complex SMARTS pattern schemes (e.g., valence or connectivity constraints, some ring constraints).
            This function works best when sampling given a list of atoms. However, sampling depends on the model's conditional probabilities,
            meaning that if the model assigns zero probability to a token, you are unlikely to see it.

        Args:
            scaffold: Scaffold (with attachment points) to decorate.
            n_samples_per_trial: Number of new molecules to generate for each randomization.
            n_trials: Number of randomizations to perform.
            do_not_fragment_further: Whether to prevent further fragmentation of the scaffold.
            sanitize: Whether to sanitize the generated molecules and ensure the scaffold is present.
            random_seed: Seed for randomization.
            n_scaff_random: Number of scaffold randomizations to try (to reposition constraints in the string and increase rollout likelihood).
                Increasing this will improve sampling, but will require more time.
            n_scaff_samples: Maximum number of samples to sample for a given scaffold from the pattern.
                Increasing this will make sure you have more diversity in the scaffold coming from the pattern
            scaff_temperature: Temperature to use when sampling valid scaffolds from the pattern. Higher temperature means more diverse scaffold
            refine: quality mode. Oversample, then
                return only valid, deduplicated molecules up to the requested count; for
                completion tasks the result is also constrained to a single connected molecule.
            kwargs: Additional arguments for the underlying generation function.

        Returns:
            List of decorated molecule sequences.
        """

        rng = random.Random(random_seed)
        n_trials = n_trials or 1
        smarts_scaffolds = [scaffold]
        if n_scaff_random and n_scaff_random > 0:
            smarts_scaffolds = PatternConstraint.randomize(
                scaffold,
                n_scaff_random,
                seed=random_seed,
            )

        all_scaffolds = {}
        scaffold_sample_count = (
            n_samples_per_trial
            if n_scaff_samples is None
            else min(n_samples_per_trial, n_scaff_samples)
        )
        for sm in smarts_scaffolds:
            cur_dec_pattern = PatternConstraint(sm, self.tokenizer, temperature=scaff_temperature)
            decorator = PatternSampler(self.model, cur_dec_pattern)
            cur_scaffolds = decorator.sample_scaffolds(
                n_samples=scaffold_sample_count,
                n_trials=1,
                random_seed=rng.randint(1, 2**32 - 1),
            )
            all_scaffolds.update(dict.fromkeys(cur_scaffolds))

        # Pattern sampling resolves atom queries to concrete tokens. Parse the
        # result as a molecule rather than preserving it as a query molecule:
        # completion requires a chemically valid molecular graph.
        parsed_scaffolds = []
        for sampled_scaffold in all_scaffolds:
            scaffold_mol = dm.to_mol(sampled_scaffold, remove_hs=False)
            if scaffold_mol is not None:
                parsed_scaffolds.append(scaffold_mol)

        total_sequences = []
        for scaffold_mol in parsed_scaffolds:
            with dm.without_rdkit_log():
                cur_sequences = self._completion(
                    fragment=scaffold_mol,
                    n_samples_per_trial=int(n_samples_per_trial / max(len(parsed_scaffolds), 1))
                    + 1,
                    n_trials=n_trials,
                    do_not_fragment_further=do_not_fragment_further,
                    sanitize=sanitize or refine,
                    random_seed=rng.randint(1, 2**32 - 1),
                    add_dot=add_dot,
                    refine=refine,
                    **kwargs,
                )
                total_sequences.extend(cur_sequences)

        rng.shuffle(total_sequences)
        if sanitize or refine:
            total_sequences = sf.utils.filter_by_substructure_constraints(total_sequences, scaffold)
            if self.verbose:
                logger.info(
                    f"After sanitization, {len(total_sequences)} / {n_samples_per_trial * n_trials} "
                    f"({len(total_sequences) * 100 / (n_samples_per_trial * n_trials):.2f}%) generated molecules are valid!"
                )

        return self._finalize_samples(
            total_sequences,
            n_samples_per_trial * n_trials,
            refine,
        )

    def de_novo_generation(
        self,
        n_samples_per_trial: int = 10,
        sanitize: bool = False,
        n_trials: Optional[int] = None,
        refine: bool = False,
        **kwargs: Optional[Dict[Any, Any]],
    ):
        """Perform de novo generation using the pretrained SAFE model.

        De novo generation is equivalent to not having any prefix.

        Args:
            n_samples_per_trial: number of new molecules to generate
            sanitize: whether to perform sanitization, aka, perform control to ensure what is asked is what is returned
            n_trials: number of randomization to perform
            refine: quality mode. Oversample, then
                return only valid, deduplicated molecules up to the requested count; for
                completion tasks the result is also constrained to a single connected molecule.
            kwargs: any argument to provide to the underlying generation function
        """
        kwargs.setdefault("how", "random")
        if kwargs["how"] != "random" and not kwargs.get("do_sample"):
            logger.warning(
                "Deterministic decoding can return repeated de novo samples; use "
                "do_sample=True or how='random' when diversity is required"
            )

        total_sequences = []
        n_trials = n_trials or 1
        candidates_per_trial = self._candidate_count(n_samples_per_trial, refine)
        for _ in tqdm(range(n_trials), disable=(not self.verbose), leave=False):
            sequences = self._generate(n_samples=candidates_per_trial, **kwargs)
            total_sequences.extend(sequences)
        total_sequences = self._decode_safe(
            total_sequences, canonical=True, remove_invalid=sanitize or refine
        )

        if sanitize and self.verbose:
            logger.info(
                f"After sanitization, {len(total_sequences)} / {n_samples_per_trial*n_trials} ({len(total_sequences)*100/(n_samples_per_trial*n_trials):.2f} %) generated molecules are valid !"
            )
        return self._finalize_samples(
            total_sequences,
            n_samples_per_trial * n_trials,
            refine,
        )

    def _find_fragment_cut(self, fragment: str, prefix_constraint: str, branching_id: str):
        """
        Perform a cut on the input fragment in such a way that it could be joined with another fragments sharing the same
        branching id.

        Args:
            fragment: fragment to cut
            prefix_constraint: prefix constraint to use
            branching_id: branching id to use
        """
        prefix_constraint = prefix_constraint.rstrip(".") + "."
        fragment = (
            fragment.replace(prefix_constraint, "", 1)
            if fragment.startswith(prefix_constraint)
            else fragment
        )
        fragments = fragment.split(".")
        i = 0
        for idx, x in enumerate(fragments):
            if branching_id in x:
                i = idx + 1
                break
        return ".".join(fragments[:i])

    def __mix_sequences(
        self,
        prefix_sequences: List[str],
        suffix_sequences: List[str],
        prefix: str,
        suffix: str,
        n_samples: int,
        mol_linker_slicer,
    ):
        """Use generated prefix and suffix sequences to form new molecules
        that will be the merging of both. This is the two step scaffold morphing and linker generation scheme
        Args:
            prefix_sequences: list of prefix sequences
            suffix_sequences: list of suffix sequences
            prefix: decoded smiles of the prefix
            suffix: decoded smiles of the suffix
            n_samples: number of samples to generate
        """
        prefix_linkers = []
        suffix_linkers = []
        prefix_query = dm.from_smarts(prefix)
        suffix_query = dm.from_smarts(suffix)

        for x in prefix_sequences:
            molecule = dm.to_mol(x)
            if molecule is not None:
                prefix_linkers.append(mol_linker_slicer(molecule, prefix_query)[1])
        for x in suffix_sequences:
            molecule = dm.to_mol(x)
            if molecule is not None:
                suffix_linkers.append(mol_linker_slicer(molecule, suffix_query)[1])
        linked = []
        linkers = dict.fromkeys(
            linker for linker in prefix_linkers + suffix_linkers if linker is not None
        )
        for linker in linkers:
            linked.extend(mol_linker_slicer.link_fragments(linker, prefix, suffix))
            linked = list(dict.fromkeys(x for x in linked if x))
            if len(linked) >= n_samples:
                break
        return linked[:n_samples]

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
        refine: bool = False,
        **kwargs,
    ):
        """Perform sentence completion using a prefix fragment

        Args:
            fragment: fragment (with attachment points)
            n_samples_per_trial: number of new molecules to generate for each randomization
            n_trials: number of randomization to perform
            do_not_fragment_further: whether to fragment the scaffold further or not
            sanitize: whether to sanitize the generated molecules
            random_seed: random seed to use
            is_safe: whether the smiles is already encoded as a safe string
            add_dot: whether to add a dot at the end of the fragments to signal to the model that we want to generate a distinct fragment.
            refine: quality mode. Oversample, then
                return only valid, deduplicated molecules up to the requested count; for
                completion tasks the result is also constrained to a single connected molecule.
            kwargs: any argument to provide to the underlying generation function
        """

        # EN: lazy programming much ?
        kwargs.setdefault("how", "random")
        if kwargs["how"] != "random" and not kwargs.get("do_sample"):
            logger.warning(
                "Deterministic completion can return repeated samples; use do_sample=True or "
                "how='random' when diversity is required"
            )

        # Step 1: convert the fragment into the relevant SAFE string format.
        # we use the provided safe encoder with the slicer that was expected

        rng = random.Random(random_seed)

        total_sequences = []
        n_trials = n_trials or 1
        candidates_per_trial = self._candidate_count(n_samples_per_trial, refine)
        for _ in tqdm(range(n_trials), disable=(not self.verbose), leave=False):
            new_seed = rng.randint(1, 2**32 - 1)
            if is_safe:
                encoded_fragment = fragment
            else:
                with dm.without_rdkit_log():
                    context_mng = (
                        sf.utils.attr_as(self.safe_encoder, "slicer", None)
                        if do_not_fragment_further
                        else suppress()
                    )
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

            if add_dot and encoded_fragment.count("(") == encoded_fragment.count(")"):
                encoded_fragment = encoded_fragment.rstrip(".") + "."

            # Quality mode also steers decoding toward a single connected
            # molecule so completions do not come back as the scaffold plus
            # spurious disconnected fragments.
            force_connected = kwargs.pop("force_connected", False) or refine
            sequences = self._generate(
                n_samples=candidates_per_trial,
                safe_prefix=encoded_fragment,
                force_connected=force_connected,
                **kwargs,
            )

            sequences = self._decode_safe(
                sequences,
                canonical=True,
                remove_invalid=sanitize or refine,
            )
            if refine:
                # Drop any residual disconnected completions before dedup/cap.
                sequences = [s for s in sequences if s is not None and "." not in s]
            total_sequences.extend(sequences)

        return self._finalize_samples(
            total_sequences,
            candidates_per_trial * n_trials,
            refine,
        )

    def _generate(
        self,
        n_samples: int = 1,
        safe_prefix: Optional[str] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        how: Optional[str] = "random",
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
        force_connected: bool = False,
        **kwargs,
    ):
        """Sample a SAFE sequence with the maintained Transformers generation paths.

        ??? note "Generation Parameters"
            SAFE maintains multinomial sampling, greedy decoding, beam search,
            beam sampling and the constrained beam path required by model-only
            linker generation. For other experimental Transformers strategies,
            call the underlying model directly.

        Args:
            n_samples: number of sequences to return
            safe_prefix: Prefix to use in sampling, should correspond to a safe fragment
            max_length: deprecated total sequence-length limit, including the prompt.
            max_new_tokens: maximum number of tokens generated after the prompt. Defaults to 100.
            how: which sampling method to use: "beam", "greedy" or "random". Can be used to control other parameters by setting defaults
            num_beams: number of beams for beam search. 1 means no beam search, unless beam is specified then max(n_samples, num_beams) is used
            do_sample: whether to perform random sampling or not, equivalent to setting random to True
            force_connected: for completion prompts, constrain decoding so the returned
                molecule is a single connected component (see
                :class:`ScaffoldConnectivityLogitsProcessor`). Requires a scaffold prefix.
            kwargs: any additional keyword argument to pass to the underlying sampling `generate`  from hugging face transformer

        Returns:
            samples: list of sampled molecules, including failed validation

        """
        pretrained_tk = self.tokenizer.get_pretrained()
        if getattr(pretrained_tk, "model_max_length") is None:
            setattr(
                pretrained_tk,
                "model_max_length",
                self._DEFAULT_MAX_LENGTH,
            )

        input_ids = safe_prefix
        if isinstance(safe_prefix, str):
            input_ids = pretrained_tk(
                safe_prefix,
                return_tensors="pt",
            )

        if how not in {None, "random", "greedy", "beam"}:
            raise ValueError("how must be one of: 'random', 'greedy', 'beam' or None")
        if kwargs.get("num_beam_groups", 1) > 1:
            raise ValueError(
                "Diverse beam search is not maintained by SAFE; call model.generate() directly"
            )
        if kwargs.get("penalty_alpha", 0) > 0:
            raise ValueError(
                "Contrastive search is not maintained by SAFE; call model.generate() directly"
            )

        num_beams = num_beams or None
        do_sample = do_sample or False

        if how == "random":
            do_sample = True

        elif how == "beam":
            num_beams = max((num_beams or 0), n_samples)

        is_greedy = how == "greedy" or (num_beams in [0, 1, None]) and do_sample is False

        kwargs["do_sample"] = do_sample
        if num_beams is not None:
            kwargs["num_beams"] = num_beams
        kwargs["return_dict_in_generate"] = True
        kwargs["num_return_sequences"] = n_samples
        if max_length is not None and max_new_tokens is not None:
            raise ValueError("Pass only one of max_length or max_new_tokens")
        if max_length is not None:
            warnings.warn(
                "max_length keeps the legacy total-length semantics and will be removed in "
                "SAFE 2.0; use max_new_tokens for a prompt-independent generation budget.",
                FutureWarning,
                stacklevel=2,
            )
            kwargs["max_length"] = max_length
        else:
            kwargs["max_new_tokens"] = 100 if max_new_tokens is None else max_new_tokens
        # ``early_stopping`` only has a meaning for beam search. Passing it to
        # multinomial or greedy generation is ignored by Transformers and
        # emits a warning for every call.
        if num_beams is not None and num_beams > 1:
            kwargs.setdefault("early_stopping", True)
        if not isinstance(input_ids, Mapping):
            input_ids = {"inputs": None}
        else:
            # Drop the tokenizer-appended EOS so generation continues from the prefix.
            for k in input_ids:
                input_ids[k] = input_ids[k][:, :-1]

        for k, v in input_ids.items():
            if torch.is_tensor(v):
                input_ids[k] = v.to(self.model.device)

        # Remove token type IDs to support model families beyond GPT-2.
        input_ids.pop("token_type_ids", None)

        if force_connected:
            if not isinstance(safe_prefix, str):
                raise ValueError("force_connected requires a scaffold prefix to complete")
            prompt_len = int(input_ids["input_ids"].shape[1])
            connectivity = ScaffoldConnectivityLogitsProcessor(pretrained_tk, prompt_len)
            existing = kwargs.get("logits_processor") or []
            kwargs["logits_processor"] = LogitsProcessorList([*existing, connectivity])

        custom_generator = None
        if kwargs.get("constraints") is not None or kwargs.get("force_words_ids") is not None:
            custom_generator = self._load_constrained_generation_backend()

        def generate(**generation_kwargs):
            active_generation_config = self.generation_config
            if active_generation_config is not None:
                active_generation_config = copy.deepcopy(active_generation_config)
                generation_kwargs = active_generation_config.update(**generation_kwargs)
                generation_kwargs["generation_config"] = active_generation_config
            if custom_generator is None:
                return self.model.generate(**generation_kwargs)
            return custom_generator(model=self.model, **generation_kwargs)

        if is_greedy:
            kwargs["num_return_sequences"] = 1
            if num_beams is not None and num_beams > 1:
                raise ValueError("Cannot set num_beams > 1 for greedy")
            # Greedy decoding has one solution; duplicate it instead of recomputing it.
            outputs = generate(
                **input_ids,
                **kwargs,
            )
            sequences = [
                pretrained_tk.decode(outputs.sequences.squeeze(), skip_special_tokens=True)
            ] * n_samples

        else:
            outputs = generate(
                **input_ids,
                **kwargs,
            )
            sequences = pretrained_tk.batch_decode(outputs.sequences, skip_special_tokens=True)
        return sequences
