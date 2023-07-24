from typing import Optional
from typing import List

import re
import fsspec
import json
from loguru import logger
from contextlib import contextmanager

from tokenizers import decoders
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, PreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
UNK_TOKEN = "[UNK]"
PADDING_TOKEN = "[PAD]"
EOS_TOKEN = SEP_TOKEN = "[SEP]"
BOS_TOKEN = CLS_TOKEN = "[CLS]"
MASK_TOKEN = "[MASK]"
TEMPLATE_SINGLE = "[CLS] $A [SEP]"
TEMPLATE_PAIR = "[CLS] $A [SEP] $B:1 [SEP]:1"
TEMPLATE_SPECIAL_TOKENS = [
    ("[CLS]", 1),
    ("[SEP]", 2),
]


@contextmanager
def attr_as(obj, field, value):
    """Temporary replace the value of an object
    Args:
        obj: object to temporary patch
        field: name of the key to change
        value: value of key to be temporary changed
    """
    old_value = getattr(obj, field, None)
    setattr(obj, field, value)
    yield
    setattr(obj, field, old_value)


class SAFESplitter:
    """Standard Splitter for SAFE string"""

    REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

    name = "smiles"

    def __init__(self, pattern: Optional[str] = None):
        # do not use this as raw strings (not r before)
        if pattern is None:
            pattern = self.REGEX_PATTERN
        self.regex = re.compile(pattern)

    def tokenize(self, line):
        """Tokenize a molecule into SMILES."""
        if isinstance(line, str):
            tokens = list(self.regex.findall(line))
            reconstruction = "".join(tokens)
            if line != reconstruction:
                logger.error(
                    f"Tokens different from sample:\ntokens {reconstruction}\nsample {line}."
                )
                raise ValueError(line)
        else:
            idxs = re.finditer(self.regex, str(line))
            tokens = [line[m.start(0) : m.end(0)] for m in idxs]
        return tokens

    def detokenize(self, chars):
        """Detokenize SAFE notation"""
        if isinstance(chars, str):
            chars = chars.split(" ")
        return "".join([x.strip() for x in chars])

    def split(self, n, normalized):
        """Perform splitting for pretokenization"""
        return self.tokenize(normalized)

    def pre_tokenize(self, pretok):
        """Pretokenize using an input pretokenizer"""
        pretok.split(self.split)


class SAFETokenizer:
    """
    Class to initialize and train a tokenizer for SAFE string
    Once trained, you can use the converted version of the tokenizer to an HuggingFace PreTrainedTokenizerFast
    """

    def __init__(
        self,
        tokenizer_type: str = "bpe",
        trainer_args=None,
        decoder_args=None,
        token_model_args=None,
    ):
        super().__init__()
        self.tokenizer_type = tokenizer_type
        self.trainer_args = trainer_args or {}
        self.decoder_args = decoder_args or {}
        self.token_model_args = token_model_args or {}
        if tokenizer_type is not None and tokenizer_type.startswith("bpe"):
            self.model = BPE(unk_token=UNK_TOKEN, **self.token_model_args)
            self.trainer = BpeTrainer(special_tokens=SPECIAL_TOKENS, **self.trainer_args)

        else:
            self.model = WordLevel(unk_token=UNK_TOKEN, **self.token_model_args)
            self.trainer = WordLevelTrainer(special_tokens=SPECIAL_TOKENS, **self.trainer_args)

        self.tokenizer = Tokenizer(self.model)
        self.tokenizer.pre_tokenizer = PreTokenizer.custom(SAFESplitter())
        self.tokenizer.post_processor = TemplateProcessing(
            single=TEMPLATE_SINGLE,
            pair=TEMPLATE_PAIR,
            special_tokens=TEMPLATE_SPECIAL_TOKENS,
        )
        self.tokenizer.decoder = decoders.BPEDecoder(**self.decoder_args)
        self.tokenizer = self.set_special_tokens(self.tokenizer)

    @classmethod
    def set_special_tokens(cls, tokenizer, bos_token=CLS_TOKEN, eos_token=SEP_TOKEN):
        """Set special tokens for a tokenizer

        Args:
            tokenizer: tokenizer for which special tokens will be set
            bos_token: Optional bos token to use
            eos_token: Optional eos token to use
        """
        tokenizer.pad_token = PADDING_TOKEN
        tokenizer.cls_token = CLS_TOKEN
        tokenizer.sep_token = SEP_TOKEN
        tokenizer.mask_token = MASK_TOKEN
        tokenizer.unk_token = UNK_TOKEN
        tokenizer.eos_token = eos_token
        tokenizer.bos_token = bos_token
        tokenizer.add_special_tokens(
            [
                PADDING_TOKEN,
                CLS_TOKEN,
                SEP_TOKEN,
                MASK_TOKEN,
                UNK_TOKEN,
                eos_token,
                bos_token,
            ]
        )
        return tokenizer

    def train(self, files: Optional[List[str]], **kwargs):
        r"""
        This is to train a new tokenizer from either a list of file or some input data

        Args
            files (str): file in which your molecules are separated by new line
            kwargs (dict): optional args for the tokenizer `train`
        """
        if isinstance(files, str):
            files = [files]
        self.tokenizer.train(files=files, trainer=self._get_trainer())

    def train_from_iterator(self, data, **kwargs):
        """Train the Tokenizer using the provided iterator.

        You can provide anything that is a Python Iterator
            * A list of sequences :obj:`List[str]`
            * A generator that yields :obj:`str` or :obj:`List[str]`
            * A Numpy array of strings

        Args:
            data (iterator): data
            kwargs: additional keyword argument for the tokenizer `train_from_iterator`
        """
        self.tokenizer.train_from_iterator(data, trainer=self.trainer, **kwargs)

    def __len__(self):
        r"""
        Returns: Gets the count of tokens in vocab along with special tokens.

        """
        return len(self.tokenizer.get_vocab().keys())

    @classmethod
    def encode(
        cls, pretrained_tokenizer: PreTrainedTokenizerFast, sample_str, ids_only=True, **kwargs
    ):
        r"""
        Encodes a given molecule string once training is done

        Args
            pretrained_tokenizer: pretrained tokenizer to use
            sample_str (str): Sample string to encode molecule
            ids_only (bool): whether to return only the ids or the encoding objet

        Returns:
            object: Returns encoded list of IDs

        """
        enc = pretrained_tokenizer.encode(sample_str, **kwargs)
        if ids_only:
            ids = enc.ids
            if ids[0] == cls.bos_token_id:
                # by convention bos token is not
                ids = ids[1:]
            return ids
        return enc

    def to_dict(self, **kwargs):
        """Convert tokenizer to dict"""
        if not self._use_white_space(self.tokenizer_type):
            # we need to do this because HuggingFace tokenizers doesnt save with custom pre-tokenizers
            with attr_as(self.tokenizer, "pre_tokenizer", Whitespace()):
                # temporary replace pre tokenizer with whitespace
                tk_data = json.loads(self.tokenizer.to_str())
                tk_data["custom_pre_tokenizer"] = True

        else:
            tk_data = json.loads(self.tokenizer.to_str())
        tk_data["tokenizer_type"] = self.tokenizer_type
        tk_data["tokenizer_attrs"] = self.tokenizer.__dict__
        return tk_data

    def save(self, file_name=None):
        r"""
        Saves the :class:`~tokenizers.Tokenizer` to the file at the given path.

        Args:
            file_name (str, optional): File where to save tokenizer
        """
        # EN: whole logic here assumes noone is going to mess with the special token
        file_name = file_name or self.file_path
        tk_data = self.to_dict()
        with fsspec.open(file_name, "w", encoding="utf-8") as OUT:
            out_str = json.dumps(tk_data, ensure_ascii=False)
            OUT.write(out_str)

    @classmethod
    def load(cls, file_name):
        """Load the current tokenizer from file"""
        with fsspec.open(file_name, "r") as OUT:
            data_str = OUT.read()
        data = json.loads(data_str)
        # EN: the rust json parser of tokenizers has a predefined structure
        # the next two lines are important
        return cls.from_dict(data)

    @classmethod
    def decode(
        cls,
        ids,
        pretrained_tokenizer: PreTrainedTokenizerFast,
        skip_special_tokens: bool = False,
        ignore_stops: bool = False,
        stop_token_ids: Optional[List[int]] = None,
    ):
        r"""
        Decodes a list of ids to molecular representation in the format in which this tokenizer was created.

        Args:
            ids: list of IDs
            pretrained_tokenizer: pretrained tokenizer to use
            skip_special_tokens: whether to skip all special tokens when encountering them
            ignore_stops: whether to ignore the stop tokens, thus decoding till the end
            stop_token_ids: optional list of stop token ids to use

        Returns:
            sequence: str representation of molecule

        """
        new_ids = ids
        if not stop_token_ids:
            stop_token_ids = [pretrained_tokenizer.token_to_id(pretrained_tokenizer.eos_token)]
        if not ignore_stops:
            new_ids = []
            # if first tokens are stop, we just remove it
            # this is because of bart essentially
            pos = 0
            if len(ids) > 1:
                while ids[pos] in stop_token_ids:
                    pos += 1
            # we only ignore when there is a list of tokens
            ids = ids[pos:]
            for pos, id in enumerate(ids):
                if int(id) in stop_token_ids:
                    break
                new_ids.append(id)
        return pretrained_tokenizer.decode(list(new_ids), skip_special_tokens=skip_special_tokens)

    def get_tokenizer(self, **kwargs):
        r"""
        Get a pretrained tokenizer from this tokenizer

        Returns:
            Returns pre-trained fast tokenizer for hugging face models.
        """
        tk = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        # now we need to add special_tokens
        tk.add_special_tokens(
            {
                "cls_token": self.tokenizer.cls_token,
                "bos_token": self.tokenizer.bos_token,
                "eos_token": self.tokenizer.eos_token,
                "mask_token": self.tokenizer.mask_token,
                "pad_token": self.tokenizer.pad_token,
                "unk_token": self.tokenizer.unk_token,
                "sep_token": self.tokenizer.sep_token,
            }
        )
        if (
            tk.model_max_length is None
            or tk.model_max_length > 1e8
            and hasattr(self.tokenizer, "model_max_length")
        ):
            tk.model_max_length = self.tokenizer.model_max_length
            setattr(
                tk,
                "model_max_length",
                getattr(self.tokenizer, "model_max_length"),
            )
        return tk
