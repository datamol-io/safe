import copy
import functools
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import torch
from tokenizers import Tokenizer
from transformers.data.data_collator import _torch_collate_batch

from safe.tokenizer import SAFETokenizer


class SAFECollator:
    """Collate function for language modelling tasks


    !!! note
        The collate function is based on the default DataCollatorForLanguageModeling in huggingface
        see: https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/data/data_collator.py
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        pad_to_multiple_of: Optional[int] = None,
        input_key: str = "inputs",
        label_key: str = "labels",
        property_key: str = "descriptors",
        include_descriptors: bool = False,
        max_length: Optional[int] = None,
    ):
        """
        Default collator for huggingface transformers in izanagi.

        Args:
            tokenizer: Huggingface tokenizer
            input_key: key to use for input ids
            label_key: key to use for labels
            property_key: key to use for properties
            include_descriptors: whether to include training on descriptors or not
            pad_to_multiple_of: pad to multiple of this value
        """

        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.input_key = input_key
        self.label_key = label_key
        self.property_key = property_key
        self.include_descriptors = include_descriptors
        self.max_length = max_length

    @functools.lru_cache()
    def get_tokenizer(self):
        """Get underlying tokenizer"""
        if isinstance(self.tokenizer, SAFETokenizer):
            return self.tokenizer.get_pretrained()
        return self.tokenizer

    def __call__(self, samples: List[Union[List[int], Any, Dict[str, Any]]]):
        """
        Call collate function

        Args:
            samples: list of examples
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        tokenizer = self.get_tokenizer()

        # examples = samples
        examples = copy.deepcopy(samples)
        inputs = [example.pop(self.input_key, None) for example in examples]
        mc_labels = (
            torch.tensor([example.pop(self.property_key, None) for example in examples]).float()
            if self.property_key in examples[0]
            else None
        )

        if "input_ids" not in examples[0] and inputs is not None:
            batch = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = tokenizer.pad(
                examples,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                max_length=self.max_length,
            )

        # If special token mask has been preprocessed, pop it from the dict.
        batch.pop("special_tokens_mask", None)
        labels = batch.get(self.label_key, batch["input_ids"].clone())
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        batch[self.label_key] = labels

        if mc_labels is not None and self.include_descriptors:
            batch.update(
                {
                    "mc_labels": mc_labels,
                    # "input_text": inputs,
                }
            )
        return batch
