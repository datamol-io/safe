from dataclasses import dataclass, field
from typing import Optional
from typing import Literal


@dataclass
class DataConvertArguments:
    """
    Configuration for tokenizer training.
    """

    canonical: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return the canonical encoding of the SAFE string."},
    )
    randomize: Optional[bool] = field(
        default=True, metadata={"help": "Whether to return a randomized SAFE string."}
    )
    seed: Optional[int] = field(
        default=None, metadata={"help": "Optional seed to use for reproducibility."}
    )
    slicer: Optional[str] = field(
        default="brics",
        metadata={"help": "Slicing algorithm, one of []'mmpa', 'brics', 'hr', 'recap']"},
    )

    split_fragments: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to split mono fragment molecules when the slicing algorithm does not work."
        },
    )
    fraction_hs: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional fraction of hydrogen addition to add to atoms with available vectorization points, when split_fragments is True."
        },
    )


@dataclass
class TokenizerTrainingArguments:
    """
    Configuration for tokenizer training.
    """

    base_tokenizer: Optional[str] = field(
        default=None, metadata={"help": "Optional base tokenizer to build new tokenizer from."}
    )
    tokenizer_type: Optional[str] = field(
        default="bpe",
        metadata={
            "help": "Tokenizer type to use. One of 'bpe' and 'word'. 'BPE' will in theory decrease token length."
        },
    )
    dataset: Optional[str] = field(
        default=None, metadata={"help": "Name or path to dataset to train tokenizer on."}
    )
    data_key: Optional[str] = field(
        default="safe", metadata={"help": "Column containing the  data to process."}
    )
    vocab_size: Optional[int] = field(
        default=1000, metadata={"help": "Target vocab size of the tokenizer."}
    )
    max_examples: Optional[int] = field(
        default=-1, metadata={"help": "Number of examples to train the tokenizer on."}
    )
    tokenizer_name: Optional[str] = field(
        default="safe", metadata={"help": "Name of new tokenizer, if we need to push it to the hub"}
    )
    push_to_hub: Optional[bool] = field(
        default=True, metadata={"help": "Push saved tokenizer to the hub."}
    )


@dataclass
class DataArguments:
    """Configuration for the data module and preparation."""

    dataset: Optional[str] = field(
        default=None, metadata={"help": "Name or path of full dataset, with split indicators."}
    )
    max_length: Optional[bool] = field(
        default=1024,
        metadata={"help": "Maximum block size for aligning tokenized text into batches."},
    )
    num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of CPU cores to use for parallel preprocessing. Default uses the maximum available."
        },
    )
    tokenized: Optional[bool] = field(
        default=False, metadata={"help": "If True the data is pretokenized."}
    )
    streaming: Optional[bool] = field(
        default=False, metadata={"help": "If True the data is streamed."}
    )


@dataclass
class ModelArguments:
    """
    Configuration for training model.
    """

    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name or path of model config. If loading from huggingface, it will be the same as the model name."
        },
    )
    tokenizer_name: Optional[str] = (
        field(
            default=None,
            metadata={
                "help": "Name or path of tokenizer. If loading from huggingface, It will be the same as the model name"
            },
        ),
    )
    output_dir: Optional[str] = field(
        default="./data/",
        metadata={
            "help": "Path to the output folder where to save the data name or path of model to be trained."
        },
    )
    optimizer: Optional[str] = (
        field(default="adamw", metadata={"help": "Optimizer to use for training."}),
    )
    group_by_length: Optional[bool] = (
        field(
            default=False,
            metadata={
                "help": "Whether to group the dataset by length when training. Usually faster."
            },
        ),
    )

    learning_rate: Optional[float] = field(
        default=1e-4, metadata={"help": "Learning rate for training."}
    )

    num_warmup_steps: Optional[int] = field(
        default=750, metadata={"help": "Number of warmup steps in the learning rate schedule."}
    )

    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "Number of steps before logging."}
    )

    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of steps before saving the model checkpoint"}
    )

    eval_steps: Optional[int] = field(
        default=750, metadata={"help": "Number of warmup steps in the learning rate schedule."}
    )
    compile_model: Optional[Literal["reduce-overheard", "default", "max-autotune"]] = field(
        default=None,
        metadata={
            "help": "Whether to compile the model and which torch.compile mode to use. Default is None. Available options are {'reduce-overhead', 'default', 'max-autotune'}"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
