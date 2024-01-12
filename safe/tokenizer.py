import contextlib
import copy
import json
import os
import re
import warnings
from typing import Any, Dict, Iterator, List, Optional, Union

import fsspec
import numpy as np
import packaging.version
import torch
from loguru import logger
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import PreTokenizer, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers import __version__ as transformers_version
from transformers.utils import (
    PushToHubMixin,
    cached_file,
    download_url,
    extract_commit_hash,
    is_offline_mode,
    is_remote_url,
    working_or_temp_dir,
)

from .utils import attr_as

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


class SAFESplitter:
    """Standard Splitter for SAFE string"""

    REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

    name = "safe"

    def __init__(self, pattern: Optional[str] = None):
        # do not use this as raw strings (not r before)
        if pattern is None:
            pattern = self.REGEX_PATTERN
        self.regex = re.compile(pattern)

    def tokenize(self, line):
        """Tokenize a safe string into characters."""
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
        """Pretokenize using an input pretokenizer object from the tokenizer library"""
        pretok.split(self.split)


class SAFETokenizer(PushToHubMixin):
    """
    Class to initialize and train a tokenizer for SAFE string
    Once trained, you can use the converted version of the tokenizer to an HuggingFace PreTrainedTokenizerFast
    """

    vocab_files_names: str = "tokenizer.json"

    def __init__(
        self,
        tokenizer_type: str = "bpe",
        splitter: Optional[str] = "safe",
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
        self.splitter = None
        if splitter == "safe":
            self.splitter = SAFESplitter()
            self.tokenizer.pre_tokenizer = PreTokenizer.custom(self.splitter)
        self.tokenizer.post_processor = TemplateProcessing(
            single=TEMPLATE_SINGLE,
            pair=TEMPLATE_PAIR,
            special_tokens=TEMPLATE_SPECIAL_TOKENS,
        )
        self.tokenizer.decoder = decoders.BPEDecoder(**self.decoder_args)
        self.tokenizer = self.set_special_tokens(self.tokenizer)

    @property
    def bos_token_id(self):
        """Get the bos token id"""
        return self.tokenizer.token_to_id(self.tokenizer.bos_token)

    @property
    def pad_token_id(self):
        """Get the bos token id"""
        return self.tokenizer.token_to_id(self.tokenizer.pad_token)

    @property
    def eos_token_id(self):
        """Get the bos token id"""
        return self.tokenizer.token_to_id(self.tokenizer.eos_token)

    @classmethod
    def set_special_tokens(
        cls,
        tokenizer: Tokenizer,
        bos_token: str = CLS_TOKEN,
        eos_token: str = SEP_TOKEN,
    ):
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

        if isinstance(tokenizer, Tokenizer):
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
        self.tokenizer.train(files=files, trainer=self.trainer)

    def __getstate__(self):
        """Getting state to allow pickling"""
        with attr_as(self.tokenizer, "pre_tokenizer", Whitespace()):
            d = copy.deepcopy(self.__dict__)
        # copy back tokenizer level attribute
        d["tokenizer_attrs"] = self.tokenizer.__dict__.copy()
        d["tokenizer"].pre_tokenizer = Whitespace()
        return d

    def __setstate__(self, d):
        """Setting state during reloading pickling"""
        use_pretokenizer = d.get("custom_pre_tokenizer")
        if use_pretokenizer:
            d["tokenizer"].pre_tokenizer = PreTokenizer.custom(SAFESplitter())
        d["tokenizer"].__dict__.update(d.get("tokenizer_attrs", {}))
        self.__dict__.update(d)

    def train_from_iterator(self, data: Iterator, **kwargs: Any):
        """Train the Tokenizer using the provided iterator.

        You can provide anything that is a Python Iterator
            * A list of sequences :obj:`List[str]`
            * A generator that yields :obj:`str` or :obj:`List[str]`
            * A Numpy array of strings

        Args:
            data: data iterator
            **kwargs: additional keyword argument for the tokenizer `train_from_iterator`
        """
        self.tokenizer.train_from_iterator(data, trainer=self.trainer, **kwargs)

    def __len__(self):
        r"""
        Gets the count of tokens in vocab along with special tokens.
        """
        return len(self.tokenizer.get_vocab().keys())

    def encode(self, sample_str: str, ids_only: bool = True, **kwargs) -> list:
        r"""
        Encodes a given molecule string once training is done

        Args:
            sample_str: Sample string to encode molecule
            ids_only: whether to return only the ids or the encoding objet

        Returns:
            object: Returns encoded list of IDs
        """
        if isinstance(sample_str, str):
            enc = self.tokenizer.encode(sample_str, **kwargs)
            if ids_only:
                return enc.ids
            return enc

        encs = self.tokenizer.encode_batch(sample_str, **kwargs)
        if ids_only:
            return [enc.ids for enc in encs]
        return encs

    def to_dict(self, **kwargs):
        """Convert tokenizer to dict"""
        # we need to do this because HuggingFace tokenizers doesnt save with custom pre-tokenizers
        if self.splitter is None:
            tk_data = json.loads(self.tokenizer.to_str())
        else:
            with attr_as(self.tokenizer, "pre_tokenizer", Whitespace()):
                # temporary replace pre tokenizer with whitespace
                tk_data = json.loads(self.tokenizer.to_str())
                tk_data["custom_pre_tokenizer"] = True
        tk_data["tokenizer_type"] = self.tokenizer_type
        tk_data["tokenizer_attrs"] = self.tokenizer.__dict__
        return tk_data

    def save_pretrained(self, *args, **kwargs):
        """Save pretrained tokenizer"""
        self.tokenizer.save_pretrained(*args, **kwargs)

    def save(self, file_name=None):
        r"""
        Saves the :class:`~tokenizers.Tokenizer` to the file at the given path.

        Args:
            file_name (str, optional): File where to save tokenizer
        """
        # EN: whole logic here assumes noone is going to mess with the special token
        tk_data = self.to_dict()
        with fsspec.open(file_name, "w", encoding="utf-8") as OUT:
            out_str = json.dumps(tk_data, ensure_ascii=False)
            OUT.write(out_str)

    @classmethod
    def from_dict(cls, data: dict):
        """Load tokenizer from dict

        Args:
            data: dictionary containing the tokenizer info
        """
        tokenizer_type = data.pop("tokenizer_type", "safe")
        tokenizer_attrs = data.pop("tokenizer_attrs", None)
        custom_pre_tokenizer = data.pop("custom_pre_tokenizer", False)
        tokenizer = Tokenizer.from_str(json.dumps(data))
        if custom_pre_tokenizer:
            tokenizer.pre_tokenizer = PreTokenizer.custom(SAFESplitter())
        mol_tokenizer = cls(tokenizer_type)
        mol_tokenizer.tokenizer = mol_tokenizer.set_special_tokens(tokenizer)
        if tokenizer_attrs and isinstance(tokenizer_attrs, dict):
            mol_tokenizer.tokenizer.__dict__.update(tokenizer_attrs)
        return mol_tokenizer

    @classmethod
    def load(cls, file_name):
        """Load the current tokenizer from file"""
        with fsspec.open(file_name, "r") as OUT:
            data_str = OUT.read()
        data = json.loads(data_str)
        # EN: the rust json parser of tokenizers has a predefined structure
        # the next two lines are important
        return cls.from_dict(data)

    def decode(
        self,
        ids: list,
        skip_special_tokens: bool = True,
        ignore_stops: bool = False,
        stop_token_ids: Optional[List[int]] = None,
    ) -> str:
        r"""
        Decodes a list of ids to molecular representation in the format in which this tokenizer was created.

        Args:
            ids: list of IDs
            skip_special_tokens: whether to skip all special tokens when encountering them
            ignore_stops: whether to ignore the stop tokens, thus decoding till the end
            stop_token_ids: optional list of stop token ids to use

        Returns:
            sequence: str representation of molecule
        """
        old_id_list = ids
        if not isinstance(ids[0], (list, np.ndarray)) and not torch.is_tensor(ids[0]):
            old_id_list = [ids]
        if not stop_token_ids:
            stop_token_ids = [self.tokenizer.token_to_id(self.tokenizer.eos_token)]

        new_ids_list = []
        for ids in old_id_list:
            new_ids = ids
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
            new_ids_list.append(new_ids)
        if len(new_ids_list) == 1:
            return self.tokenizer.decode(
                list(new_ids_list[0]), skip_special_tokens=skip_special_tokens
            )
        return self.tokenizer.decode_batch(
            list(new_ids_list), skip_special_tokens=skip_special_tokens
        )

    def get_pretrained(self, **kwargs) -> PreTrainedTokenizerFast:
        r"""
        Get a pretrained tokenizer from this tokenizer

        Returns:
            Returns pre-trained fast tokenizer for hugging face models.
        """
        with attr_as(self.tokenizer, "pre_tokenizer", Whitespace()):
            tk = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        tk._tokenizer.pre_tokenizer = self.tokenizer.pre_tokenizer
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

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        create_pr: bool = False,
        safe_serialization: bool = False,
        **deprecated_kwargs,
    ) -> str:
        """
        Upload the tokenizer to the 🤗 Model Hub.

        Args:
            repo_id: The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir: Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message: Message to commit while pushing. Will default to `"Upload {object}"`.
            private: Whether or not the repository created should be private.
            token: The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size: Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`).
            create_pr: Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization: Whether or not to convert the model weights in safetensors format for safer serialization.
        """
        use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        repo_path_or_name = deprecated_kwargs.pop("repo_path_or_name", None)
        if repo_path_or_name is not None:
            # Should use `repo_id` instead of `repo_path_or_name`. When using `repo_path_or_name`, we try to infer
            # repo_id from the folder path, if it exists.
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead.",
                FutureWarning,
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_path_or_name` are both specified. Please set only the argument `repo_id`."
                )
            if os.path.isdir(repo_path_or_name):
                # repo_path: infer repo_id from the path
                repo_id = repo_id.split(os.path.sep)[-1]
                working_dir = repo_id
            else:
                # repo_name: use it as repo_id
                repo_id = repo_path_or_name
                working_dir = repo_id.split("/")[-1]
        else:
            # Repo_id is passed correctly: infer working_dir from it
            working_dir = repo_id.split("/")[-1]

        # Deprecation warning will be sent after for repo_url and organization
        repo_url = deprecated_kwargs.pop("repo_url", None)
        organization = deprecated_kwargs.pop("organization", None)

        repo_id = self._create_repo(
            repo_id, private, token, repo_url=repo_url, organization=organization
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            with contextlib.suppress(Exception):
                self.save_pretrained(
                    work_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization
                )

            self.save(os.path.join(work_dir, self.vocab_files_names))

            return self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        return_fast_tokenizer: Optional[bool] = False,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        r"""
        Instantiate a [`~tokenization_utils_base.PreTrainedTokenizerBase`] (or a derived class) from a predefined
        tokenizer.

        Args:
            pretrained_model_name_or_path:
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
                  file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
                  `./my_model_directory/vocab.txt`.
            cache_dir: Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
                standard cache should not be used.
            force_download: Whether or not to force the (re-)download the vocabulary files and override the cached versions if they exist.
            proxies: A dictionary of proxy servers to use by protocol or endpoint, e.g.,
                `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            token: The token to use as HTTP bearer authorization for remote files.
                If `True`, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            local_files_only: Whether or not to only rely on local files and not to attempt to download any files.
            return_fast_tokenizer: Whether to return fast tokenizer or not.

        Examples:
        ``` py
            # We can't instantiate directly the base class *PreTrainedTokenizerBase* so let's show our examples on a derived class: BertTokenizer
            # Download vocabulary from huggingface.co and cache.
            tokenizer = SAFETokenizer.from_pretrained("datamol-io/safe-gpt")

            # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
            tokenizer = SAFETokenizer.from_pretrained("./test/saved_model/")

            # If the tokenizer uses a single vocabulary file, you can point directly to this file
            tokenizer = BertTokenizer.from_pretrained("./test/saved_model/tokenizer.json")
        ```
        """
        resume_download = kwargs.pop("resume_download", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        subfolder = kwargs.pop("subfolder", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        user_agent = {
            "file_type": "tokenizer",
            "from_auto_class": from_auto_class,
            "is_fast": "Fast" in cls.__name__,
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        os.path.isdir(pretrained_model_name_or_path)
        file_path = None
        if os.path.isfile(pretrained_model_name_or_path):
            file_path = pretrained_model_name_or_path
        elif is_remote_url(pretrained_model_name_or_path):
            file_path = download_url(pretrained_model_name_or_path, proxies=proxies)

        else:
            # EN: remove this when transformers package has uniform API
            cached_file_extra_kwargs = {"use_auth_token": token}
            if packaging.version.parse(transformers_version) >= packaging.version.parse("5.0"):
                cached_file_extra_kwargs = {"token": token}
            # Try to get the tokenizer config to see if there are versioned tokenizer files.
            resolved_vocab_files = cached_file(
                pretrained_model_name_or_path,
                cls.vocab_files_names,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                subfolder=subfolder,
                user_agent=user_agent,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                _commit_hash=commit_hash,
                **cached_file_extra_kwargs,
            )
            commit_hash = extract_commit_hash(resolved_vocab_files, commit_hash)
            file_path = resolved_vocab_files

        if not os.path.isfile(file_path):
            logger.info(
                f"Can't load the following file: {file_path} required for loading the tokenizer"
            )

        tokenizer = cls.load(file_path)
        if return_fast_tokenizer:
            return tokenizer.get_pretrained()
        return tokenizer


def split(safe_str: str):
    """Split a safe string into a list of character.

    !!! note
        It's recommended to use a trained tokenizer (e.g `SAFETokenizer`) when building deeplearning models

    Args:
        safe_str: input safe string to split
    """

    splitter = SAFESplitter()
    return splitter.tokenize(safe_str)
