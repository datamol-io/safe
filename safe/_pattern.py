from typing import Union
from typing import List
from typing import Optional
import re
import datamol as dm
import torch
import numpy as np
import safe as sf
import torch.nn.functional as F
import transformers

from loguru import logger
from tqdm.auto import tqdm
import contextlib


class PatternConstraint:
    """
    Sampling decorator for pretrained SAFE models.
    This implementation is inspired by the Sanofi decorator with:
        1. new generalization to different tokenizers
        2. support for a subset of smarts notations
        3. speed improvements by dropping unnecessary steps

    !!! note
        For smarts based constraints, it's important to understand that the constraints
        are strong sampling suggestions and not necessarily the final result, meaning that they can
        fail.

    !!! warning
        Ring constraints should be interpreted as "Extended Ring System" constraints.
        Thus [r6] means an atom in an environment of 6 atoms and more that contains a ring system,
        instead of an atom in a ring of size 6.
    """

    ATTACHMENT_POINT_TOKEN = "\\*"
    ATTACHMENT_POINTS = [
        # parse any * not preceeded by "[" or ":" and not followed by "]" or ":" as attachment
        r"(?<![:\[]){0}(?![:\]])".format(ATTACHMENT_POINT_TOKEN),
        # fix some edge cases of the above for attachment in the "()" environment or in SMARTS notation
        r"(?<![\[(\[\w)]){0}".format(ATTACHMENT_POINT_TOKEN),
        # parse attachment in optionally mapped isotope smarts atoms, e.g., [1*:3]
        r"\[(\d*){0}:?(\d*)\]".format(ATTACHMENT_POINT_TOKEN),
    ]
    IS_CONSTRAINT = r"\[[^\[]*:.*\]$"
    HAS_RING_TOKEN = r"\[(.*)(r\d*)([^\[]*)\]"

    _MIN_PROBS = 1e-32

    def __init__(
        self,
        scaffold: str,
        tokenizer,
        branch_opener: str = "(",
        branch_closer: str = ")",
        min_linker_size: int = 5,
        min_ring_size: int = 3,
        force_constraint_sample: bool = True,
        temperature: float = 1.0,
    ):
        """
        Initialize the PatternConstraint with a scaffold and tokenizer.

        Args:
            scaffold: The input scaffold for decoration.
            tokenizer: The tokenizer to use for encoding the scaffold.
            branch_opener: Character to represent branch openers.
            branch_closer: Character to represent branch closers.
            min_linker_size: Minimum size of linkers.
            min_ring_size: Minimum size of rings.
            force_constraint_sample: Whether to force sampling constraints.
            temperature: temperature to apply when sampling scaffold from patterns. Higher temperature will generate more diverse scaffolds
        """
        self.input_scaffold = scaffold
        self.temperature = temperature
        self.scaffold = self._prepare_scaffold()
        self.tokenizer = tokenizer
        self.branch_opener = branch_opener
        self.branch_closer = branch_closer
        self.min_linker_size = min_linker_size
        self.min_ring_size = min_ring_size
        self.force_constraint_sample = force_constraint_sample
        self.token_masks = {}
        self.actions = {}
        self.linker_size = {}
        self.tokens = None
        self.ids = None
        self.branch_opener_id = None
        self.branch_closer_id = None
        self._is_initialized = False
        self.unknown_token_map = {}
        self.ring_token_ids = self._find_ring_tokens()

        # Initialize the scaffold decoration process with the tokenizer
        self._initialize()

    def _logprobs_to_probs(self, logprobs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Convert logits to probabilities."""

        logprobs = logprobs / self.temperature
        if mask is None:
            mask = torch.ones_like(logprobs, dtype=torch.bool)
        log_probs = torch.full_like(logprobs, float("-inf"))
        log_probs[mask] = torch.masked_select(logprobs, mask)
        if not self.force_constraint_sample:
            return log_probs.exp()
        inf_masked = torch.bitwise_and(mask, torch.isinf(log_probs))

        values = torch.tensor(self._MIN_PROBS).log().to(mask.device)
        log_probs = log_probs.masked_fill(inf_masked, values)

        probs = log_probs.exp()

        # Normalize probs so that the sum of non-masked probabilities is 1
        probs = probs * mask  # Zero out masked elements
        probs_sum = probs.sum(dim=-1, keepdim=True)  # Sum only the non-masked elements
        return probs / probs_sum  # Normalize

    @classmethod
    def randomize(cls, scaffold: str, n: int = 10):
        """Randomize a scaffold for decoration."""
        random_mol = (
            dm.from_smarts(scaffold) if not isinstance(scaffold, dm.Mol) else dm.copy_mol(scaffold)
        )
        out = set()
        while not out or n > 0:
            random_mol = dm.randomize_atoms(random_mol)
            try:
                out.add(dm.to_smarts(random_mol))
            except Exception as e:
                logger.error(e)
            n -= 1
        return out

    def _find_ring_tokens(self):
        """Find all possible ring tokens in the vocab."""
        ring_token_ids = []
        for tk, tk_ids in self.tokenizer.tokenizer.get_vocab().items():
            try:
                _ = int(tk.lstrip("%"))
                ring_token_ids.append(tk_ids)
            except ValueError:
                pass
        return ring_token_ids

    def _prepare_scaffold(self):
        """Prepare scaffold for decoration."""
        return self.input_scaffold

    def is_branch_closer(self, pos_or_token: Union[int, str]):
        """Check whether a token is a branch closer."""
        if isinstance(pos_or_token, int):
            return self.tokens[pos_or_token] == self.branch_closer
        return pos_or_token == self.branch_closer

    def is_branch_opener(self, pos_or_token: Union[int, str]):
        """Check whether a token is a branch opener."""
        if isinstance(pos_or_token, int):
            return self.tokens[pos_or_token] == self.branch_opener
        return pos_or_token == self.branch_opener

    def __len__(self):
        """Get length of the tokenized scaffold."""
        if not self._is_initialized:
            raise ValueError("Decorator is not initialized yet")
        return len(self.tokens)

    def _initialize(self):
        """
        Initialize the current scaffold decorator object with the scaffold object.
        The initialization will also set and validate the vocab object to use for the scaffold decoration,
        """
        self._is_initialized = False
        pretrained_tokenizer = self.tokenizer.get_pretrained()
        encoding_tokens = [
            token
            for token, _ in pretrained_tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(
                self.scaffold
            )
        ] + [pretrained_tokenizer.eos_token]
        encoding_token_ids = [
            pretrained_tokenizer.convert_tokens_to_ids(x) for x in encoding_tokens
        ]
        # encodings.tokens contains BOS and EOS

        self.branch_opener_id = self.tokenizer.tokenizer.token_to_id(self.branch_opener)
        self.branch_closer_id = self.tokenizer.tokenizer.token_to_id(self.branch_closer)
        linker_size = {}
        # convert the full vocab into mol constraints
        vocab_as_constraints = [
            self._parse_token_as_mol(self.tokenizer.tokenizer.id_to_token(i))
            for i in range(len(self.tokenizer))
        ]
        token_masks = {}
        actions = {}
        tokens = []
        ids = []
        all_tokens = self.tokenizer.tokenizer.get_vocab().keys()
        unk_token_id = pretrained_tokenizer.unk_token_id
        unknown_token_map = {}
        # we include the stop token
        for pos in range(len(encoding_token_ids)):
            token_id = encoding_token_ids[pos]
            token = encoding_tokens[pos]

            # if it is not an unknown token, then it can just be rollout
            # note that we are not using all special tokens, just unknown
            # and as such, you would need to have a well defined vocab
            cur_mask = torch.ones(len(vocab_as_constraints), dtype=torch.bool)
            constraints = None
            ring_token = None
            if token_id == unk_token_id:
                # this here can be one of the case we want:
                # we need to check if the token is a ring and whether it has other constraints.
                ring_match = re.match(self.HAS_RING_TOKEN, token)
                if ring_match and token.count("r") != 1:
                    raise ValueError("Multiple ring constraints in a single token is not supported")
                if ring_match:
                    ring_size = ring_match.group(2).strip("r:")
                    ring_size = int(ring_size) if ring_size else self.min_ring_size
                    linker_size[pos - 1] = ring_size
                    # since we have filled the ring constraints already, we need to remove it from the token format
                    ring_token = self._remove_ring_constraint(token)
                if self._is_attachment(token) or (
                    ring_token is not None and self._is_attachment(ring_token)
                ):
                    # the mask would be handled by the attachment algorithm in decorate
                    actions[token] = "attach"
                    unknown_token_map[pos] = sf.utils.standardize_attach(token)
                elif self._is_constraint(token, all_tokens):
                    actions[token] = "constraint"
                    constraints = self._parse_token_as_mol(token)
                    if constraints is not None:
                        cur_mask = self._mask_tokens(constraints, vocab_as_constraints)
            else:
                # this means that we need to sample the exact token
                # and disallow all the other
                cur_mask = torch.zeros(len(vocab_as_constraints), dtype=torch.bool)
                cur_mask[token_id] = 1
            token_masks[token] = cur_mask
            tokens.append(token)
            ids.append(token_id)
        self.linker_size = linker_size
        self.token_masks = token_masks
        self.actions = actions
        self.tokens = tokens
        self.ids = ids
        self._is_initialized = True
        self.unknown_token_map = unknown_token_map
        self.ring_token_ids = self._find_ring_tokens()

    def _is_attachment(self, token: str):
        """Check whether a token is should be an attachment or not"""
        # What I define as attachment is a token that is not a constraint and not a ring
        # basically the classic "[*]" written however you like it
        return any(re.match(attach_regex, token) for attach_regex in self.ATTACHMENT_POINTS)

    def _is_constraint(self, token: str, vocab_list: Optional[List[str]] = None):
        """Check whether a token is a constraint
        Args:
            token: token to check whether
            vocab: optional vocab to check against
        """
        if vocab_list is None:
            vocab_list = []
        tk_constraints = re.match(self.IS_CONSTRAINT, token) or token not in vocab_list
        return tk_constraints is not None

    def _remove_ring_constraint(self, token):
        """Remove ring constraints from a token"""
        token = re.sub(r"((&|,|;)?r\d*)*(&|,|;)?", r"\3", token)
        token = re.sub(r"(\[[&;,]?)(.*)([&,;]?\])", r"[\2]", token)
        if token == "[]":
            return "[*]"
        return token

    def _parse_token_as_mol(self, token):
        """
        Parse a token as a valid molecular pattern
        """
        tk_mol = None
        with dm.without_rdkit_log():
            try:
                tk_mol = dm.from_smarts(token)
            except:
                tk_mol = dm.to_mol(token, sanitize=True)
            # didn't work, try with second strategy
            if tk_mol is None:
                tk_mol = dm.to_mol(token, sanitize=False)
                try:
                    tk_mol = dm.from_smarts(dm.smiles_as_smarts(tk_mol))
                except:
                    tk_mol = None
        return tk_mol

    def _mask_tokens(self, constraint: List[dm.Mol], vocab_mols: List[dm.Mol]):
        """Mask the prediction to enforce some constraints

        Args:
            constraint: constraint found in the scaffold
            vocab_mols: list of mol queries (convertible ones) from the vocab

        Returns:
            mask: mask for valid tokens that match constraints. 1 means keep, 0 means mask
        """
        mask = torch.zeros(len(vocab_mols), dtype=torch.bool)
        with dm.without_rdkit_log():
            for ind, tk_mol in enumerate(vocab_mols):
                if tk_mol is not None:
                    with contextlib.suppress(Exception):
                        mask[ind] = int(tk_mol.HasSubstructMatch(constraint))
        return mask


class PatternSampler:
    """
    Implements a pattern-constrained sequence sampler for Autoregressive transformer models using a PatternConstraint.

    Args:
        model: Pretrained model used for generation.
        pattern_decorator: The PatternConstraint object that provides the scaffold and sampling constraints.
        min_linker_size: Minimum size of the linker.
        max_steps_to_eos: Maximum steps to end-of-sequence token.
        max_length: Maximum length of the generated sequences.
    """

    def __init__(
        self,
        model,
        pattern_decorator,
        min_linker_size: int = 3,
        max_steps_to_eos: int = 50,
        max_length: int = 128,
    ):
        self.model = model
        self.pattern_decorator = pattern_decorator
        self.tokenizer = self.pattern_decorator.tokenizer.get_pretrained()
        self.min_linker_size = min_linker_size
        self.max_steps_to_eos = max_steps_to_eos
        self.model.eval()
        self.max_length = max_length

    def nll_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Custom Negative Log Likelihood (NLL) loss that returns loss per example.

        Args:
            inputs (torch.Tensor): Log probabilities of each class, shape (batch_size, num_classes).
            targets (torch.Tensor): Target class indices, shape (batch_size).

        Returns:
            torch.Tensor: Loss for each example, shape (batch_size).
        """
        target_expanded = (
            torch.zeros(inputs.size()).cuda()
            if torch.cuda.is_available()
            else torch.zeros(inputs.size())
        )
        target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
        loss = target_expanded * inputs
        return torch.sum(loss, dim=1)

    def sample_scaffolds(
        self, n_samples: int = 100, n_trials: int = 1, random_seed: Optional[int] = None
    ):
        """
        Sample a batch of sequences based on the scaffold provided by the PatternConstraint.

        Args:
            n_samples: Number of sequences to sample.
            n_trials: Number of sampling trials to perform.
            random_seed: Seed for random number generation.

        Returns:
            List[str]: List of sampled sequences as strings.
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)

        sampled_mols = []
        for _ in tqdm(range(n_trials), leave=False):
            generated_sequences, *_ = self._generate(n_samples)
            sampled_mols.extend(
                [
                    self._as_scaffold(self.tokenizer.decode(seq, skip_special_tokens=False))
                    for seq in generated_sequences
                ]
            )
        return sampled_mols

    def _as_scaffold(self, scaff: str) -> str:
        """
        Converts the generated sequence to a valid scaffold by replacing unknown tokens.

        Args:
            scaff: The generated sequence string.

        Returns:
            str: The scaffold string with unknown tokens replaced.
        """
        out = scaff.replace(self.tokenizer.eos_token, "")
        splitted_out = [
            token for token, _ in self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(out)
        ]
        for pos, query in self.pattern_decorator.unknown_token_map.items():
            splitted_out[pos] = query
        return "".join(splitted_out)

    def _generate(self, batch_size: int, max_length: Optional[int] = None):
        """
        Generate sequences with custom constraints using the model and PatternConstraint.

        Args:
            batch_size: Number of sequences to generate.
            max_length: Maximum length of the sequence.

        Returns:
            Tuple: Generated sequences, log probabilities, and entropies.
        """
        sequences = []
        if max_length is None:
            max_length = self.max_length

        start_token = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=bool)
        log_probs = torch.zeros(batch_size)
        entropy = torch.zeros(batch_size)

        if torch.cuda.is_available():
            log_probs = log_probs.cuda()
            entropy = entropy.cuda()
            start_token = start_token.cuda()
            finished = finished.cuda()

        max_dec_steps = max_length - len(self.pattern_decorator)
        if max_dec_steps < 0:
            raise ValueError("Step size negative due to scaffold being longer than max_length")

        input_ids = start_token
        trackers = torch.zeros(batch_size, dtype=torch.int)  # Tracks the position in the scaffold

        for step in range(max_length):
            current_tokens = [self.pattern_decorator.tokens[index] for index in trackers]
            action_i = [
                self.pattern_decorator.actions.setdefault(
                    self.pattern_decorator.tokens[index], "roll"
                )
                for index in trackers
            ]

            # Pass through model
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            log_prob = torch.log_softmax(logits, dim=-1)
            probs = log_prob.exp()

            decoder_input = torch.multinomial(probs, num_samples=1).squeeze(1).view(-1)

            for i in range(batch_size):
                if action_i[i] == "constraint":
                    mask = self.pattern_decorator.token_masks[current_tokens[i]].to(
                        input_ids.device
                    )
                    prob_i = self.pattern_decorator._logprobs_to_probs(log_prob[i, :], mask)
                    if prob_i.sum() == 0:
                        random_choice = torch.nonzero(mask.squeeze()).squeeze().cpu().numpy()
                        decoder_input[i] = np.random.choice(random_choice)
                    else:
                        decoder_input[i] = torch.multinomial(prob_i, num_samples=1).view(-1)
                    trackers[i] += int(decoder_input[i] != self.tokenizer.eos_token_id)
                else:
                    decoder_input[i] = self.pattern_decorator.ids[trackers[i].item()]
                    trackers[i] += int(decoder_input[i] != self.tokenizer.eos_token_id)

            sequences.append(decoder_input.unsqueeze(-1))
            input_ids = torch.cat((input_ids, decoder_input.unsqueeze(-1)), dim=1)
            log_probs += self.nll_loss(log_prob, decoder_input)
            entropy += -torch.sum(log_prob * probs, dim=-1)

            eos_sampled = (decoder_input == self.tokenizer.eos_token_id).bool()
            finished = torch.ge(finished + eos_sampled, 1)
            if torch.prod(finished) == 1:
                break
        sequences = torch.cat(sequences, dim=1)
        return sequences, log_probs, entropy
