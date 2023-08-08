from typing import Optional

from collections.abc import Mapping
from dataclasses import dataclass, field
from transformers import AutoTokenizer, HfArgumentParser
from safe.tokenizer import SAFETokenizer
from safe.trainer.data_utils import batch_iterator, get_dataset


@dataclass
class TokenizerTrainingArguments:
    """
    Configuration for tokenizer training.
    """
    tokenizer_type: Optional[str] = field(
        default="bpe", metadata={"help": "Type of the tokenizer to train."}
    )
    base_tokenizer: Optional[str] = field(
        default=None, metadata={"help": "Optional base tokenizer to you. Otherwise, the tokenizer will be learnt from scratch using the safe tokenizer."}
    )
    splitter: Optional[str] = field(
        default=None, metadata={"help": "Presplitter to use to train SAFE tokenizer."}
    )    
    dataset: str = field(
        default=None, metadata={"help": "Path to the dataset to load for training the tokenizer."}
    )
    text_column: Optional[str] = field(default="inputs", metadata={"help": "Column containing text data to process."})
    vocab_size: Optional[int] = field(default=1000, metadata={"help": "Target vocab size of the final tokenizer."})
    batch_size: Optional[int] = field(default=100, metadata={"help": "Batch size for training the tokenizer."})
    n_examples: Optional[int] = field(
        default=None, metadata={"help": "Number of examples to train the tokenizer on."}
    )
    tokenizer_name: Optional[str] = field(default="safe", metadata={"help": "Name of new tokenizer."})
    outfile: Optional[str] = field(default=None, metadata={"help": "Path to the local save of the trained tokenizer"})
    all_split: Optional[bool] = field(default=False, metadata={"help": "Whether to use all the splits or just the train split if only that is available."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Whether to push saved tokenizer to the hub."})



if __name__ == "__main__":
    # Configuration
    parser = HfArgumentParser(TokenizerTrainingArguments)
    args = parser.parse_args()
    dataset = get_dataset(args.dataset, streaming=True, tokenize_column=args.text_column)
    # Training and saving
    if isinstance(dataset, Mapping) and not args.all_split:
        dataset = dataset["train"]

    dataset_iterator = batch_iterator(
        dataset, batch_size=args.batch_size, n_examples=args.n_examples, column=args.text_column
    )

    if args.base_tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
        tokenizer = tokenizer.train_new_from_iterator(dataset_iterator, vocab_size=args.vocab_size)
    else:
        tokenizer = SAFETokenizer(tokenizer_type=args.tokenizer_type, splitter=args.splitter, trainer_args={'vocab_size':args.vocab_size})
        tokenizer.train_from_iterator(dataset_iterator)
        tokenizer_name = f"{args.tokenizer_name}-{args.tokenizer_type}-{args.vocab_size}" 
        # also save locally to the outfile specified 
        if args.outfile is not None:
            tokenizer.save(args.outfile)
        tokenizer = tokenizer.get_pretrained()
    tokenizer.save_pretrained(tokenizer_name, push_to_hub=args.push_to_hub)
    


