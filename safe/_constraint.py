from typing import Optional
from collections import Counter
from transformers.generation import LogitsProcessor

import torch
import safe as sf


def is_subtensor(large_tensor: torch.Tensor, small_tensor: torch.Tensor):
    """Check if input tensor is a sub-tensor of a larger tensor.
    Return the number of occurence per batch element.
    Args:
        large_tensor: A tensor of shape [batch_size, large_length, ...]
        small_tensor: A tensor of shape [small_length, ...]
    """
    large_windowed = large_tensor.unfold(1, small_tensor.size(0), 1)
    return (large_windowed == small_tensor).all(dim=-1).sum(dim=-1)


class LinkingLogitProcessor(LogitsProcessor):
    """Logits processor for fragment linking"""

    def __init__(
        self,
        fragments: str,
        tokenizer: sf.SAFETokenizer,
        fragment_splitter: str = ".",
        max_link_tokens: Optional[int] = None,
        bias_values: Optional[float] = 0,
    ):
        max_link_tokens = max_link_tokens or int(1e9)
        if not isinstance(max_link_tokens, int) or max_link_tokens < 0:
            raise ValueError(
                f"`max_link_tokens` has to be a non-negative integer, but is {max_link_tokens}"
            )

        self.max_link_tokens = max_link_tokens
        self.bias_value = bias_values
        self.eos_token_id = [tokenizer.eos_token_id]
        self.tokenizer = tokenizer
        self.fragments = fragments
        self.fragment_max_length = len(tokenizer.encode(fragments, add_special_tokens=False))
        missing_closure = self._get_missing_closure(fragments)
        self.fragment_splitter_token_ids = [
            tk_id
            for tk, tk_id in tokenizer.tokenizer.get_vocab().items()
            if fragment_splitter in tk
        ]
        self.missing_closure_map = {}
        for closure in missing_closure:
            self.missing_closure_map[closure] = tokenizer.encode(closure, add_special_tokens=False)

    def _get_missing_closure(self, substr: str):
        """Get unclosed closure in a fragment string

        Args:
            substr: A string representing a fragment
        """
        missing_closure = Counter(sf.SAFEConverter._find_branch_number(substr))
        missing_closure = [f"{str(x)}" for x in missing_closure if missing_closure[x] % 2 == 1]
        return [f"%{x}" if len(x) > 1 else x for x in missing_closure]

    def _evaluate(self, input_ids: torch.LongTensor):
        """Evaluate if the condition that we are requiring are met already"""

        missing_completion = {}
        closure_was_seen = {cl: False for cl in self.missing_closure_map}
        for cl, cl_tk in self.missing_closure_map.items():
            if not closure_was_seen.get(cl, False):
                matches = is_subtensor(input_ids, torch.tensor(cl_tk).to(input_ids))
                completed = torch.where(matches % 2 == 0)[0]
                # we have fullfilled the completion for this closure everywhere
                # we prevent reopening completion when we have already seen it
                # this means that if the model open a new bracket we don't monitor it
                if completed.shape[0] == input_ids.shape[0]:
                    closure_was_seen[cl] = True
                missing_completion[cl] = completed
        return missing_completion, closure_was_seen

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        bias = torch.zeros_like(scores)

        # if we haven't exceed the max length already, let's continue searching
        if not cur_len - self.fragment_max_length > self.max_link_tokens:
            current_build = self.tokenizer.get_pretrained().batch_decode(
                input_ids[:, len(self.fragments) + 1 :]
            )
            added_build = [self._get_missing_closure(x) for x in current_build]
            added_build = [i for i, x in enumerate(added_build) if len(x) > 0]

            # 1. do not sample eos as long as we don't have min_link_tokens
            # 2. do not sample new fragment as long as there is not promise to link
            # existing fragments
            # we consider two cases:
            # a) either all fragments linking closure have been seen then we continue as if nothing
            # b) some fragment closure have been seen. Then, as long as another opening has been added
            # we can continue
            # c) otherwise we prevent opening new fragment '.'
            missing_completion, closure_was_seen = self._evaluate(input_ids)
            for cl, cl_ids in self.missing_closure_map.items():
                if not closure_was_seen.get(cl, False):
                    bias[:, cl_ids] = self.bias_value
            if not all(closure_was_seen.values()):
                for _, ids in missing_completion.items():
                    ids = torch.unique(torch.cat([ids, torch.tensor(added_build).to(ids)], dim=-1))
                    for i in self.fragment_splitter_token_ids:  # + self.eos_token_id:
                        scores[ids, i] = -float("inf")
        return scores + bias
