from typing import Optional
from typing import Callable
import upath
import datasets


def get_dataset(
    data_path,
    name: Optional[str] = None,
    tokenizer: Optional[Callable] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = True,
    use_auth_token: bool = False,
    tokenize_column: Optional[str] = "inputs",
    block_size: Optional[int] = None,
):
    """Get the datasets from the config file"""
    raw_datasets = {}
    if data_path is not None:
        data_path = upath.UPath(str(data_path))

        if data_path.exists():
            # the we need to load from disk
            data_path = str(data_path)
            try:
                raw_datasets = datasets.load_dataset(data_path, streaming=True, cache_dir=cache_dir)
            except:
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
    if tokenizer is None:
        return raw_datasets

    block_size = tokenizer.tokenizer.model_max_length or 1024

    def tokenize(row):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            row[tokenize_column],
            truncation=True,
            max_length=block_size,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < block_size
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        result["mc_labels"] = result["descriptors"]
        return result

    columns_to_remove = [
        x for x in raw_datasets["train"].column_names if x != tokenize_column and "label" not in x
    ]
    return raw_datasets.map(tokenize, batched=True, remove_columns=columns_to_remove)
