from typing import Optional
from typing import Callable
from collections.abc import Mapping
from tqdm.auto import tqdm
from functools import partial
import itertools
import upath
import datasets
from safe.tokenizer import SAFETokenizer


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def tokenize_fn(
    row,
    tokenizer,
    tokenize_column: str = "inputs",
    max_length: Optional[int] = None,
    padding: bool = False,
):
    """Perform the tokenization of a row
    Args:
        row: row to tokenize
        tokenizer: tokenizer to use
        tokenize_column: column to tokenize
        max_length: maximum size of the tokenized sequence
        padding: whether to pad the sequence
    """
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast

    fast_tokenizer = (
        tokenizer.get_pretrained() if isinstance(tokenizer, SAFETokenizer) else tokenizer
    )

    return fast_tokenizer(
        row[tokenize_column],
        truncation=(max_length is not None),
        max_length=max_length,
        padding=padding,
        return_tensors=None,
    )


def batch_iterator(datasets, batch_size=100, n_examples=None, column="inputs"):
    if isinstance(datasets, Mapping):
        datasets = list(datasets.values())

    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]

    for dataset in datasets:
        iter_dataset = iter(dataset)
        if n_examples is not None and n_examples > 0:
            for _ in tqdm(range(0, n_examples, batch_size)):
                out = [next(iter_dataset)[column] for _ in range(batch_size)]
                yield out
        else:
            for out in tqdm(iter(partial(take, batch_size, iter_dataset), [])):
                yield [x[column] for x in out]


def get_dataset(
    data_path,
    name: Optional[str] = None,
    tokenizer: Optional[Callable] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = True,
    use_auth_token: bool = False,
    tokenize_column: Optional[str] = "inputs",
    property_column: Optional[str] = "descriptors",
    max_length: Optional[int] = None,
):
    """Get the datasets from the config file"""
    raw_datasets = {}
    if data_path is not None:
        data_path = upath.UPath(str(data_path))

        if data_path.exists():
            # the we need to load from disk
            data_path = str(data_path)
            # for some reason, the datasets package is not able to load the dataset
            # because the split where not originally proposed
            raw_datasets = datasets.load_from_disk(data_path)

            if streaming:
                if isinstance(raw_datasets, datasets.DatasetDict):
                    raw_datasets = datasets.IterableDatasetDict(
                        {k: dt.to_iterable_dataset() for k, dt in raw_datasets.items()}
                    )
                else:
                    raw_datasets = raw_datasets.to_iterable_dataset()

        else:
            raw_datasets = datasets.load_dataset(
                data_path,
                name=name,
                cache_dir=cache_dir,
                use_auth_token=True if use_auth_token else None,
                streaming=streaming,
            )
    # that means we need to return a tokenized version of the dataset

    raw_datasets = raw_datasets.rename_column(property_column, "mc_labels")
    columns_to_remove = [
        x
        for x in raw_datasets["train"].column_names
        if x not in [tokenize_column, "mc_labels"] and "label" not in x
    ] or None

    if tokenizer is None:
        if columns_to_remove is not None:
            raw_datasets = raw_datasets.remove_columns(columns_to_remove)
        return raw_datasets

    return raw_datasets.map(
        partial(
            tokenize_fn,
            tokenizer=tokenizer,
            tokenize_column=tokenize_column,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=columns_to_remove,
    )

