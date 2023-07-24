from typing import Optional
import upath
import datasets


def get_dataset(
    data_path,
    name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    use_auth_token: bool = False,
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
    return raw_datasets


# def tokenize(prompt, add_eos_token=True):
#     # there's probably a way to do this with the tokenizer settings
#     # but again, gotta move fast
#     result = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=cutoff_len,
#         padding=False,
#         return_tensors=None,
#     )
#     if (
#         result["input_ids"][-1] != tokenizer.eos_token_id
#         and len(result["input_ids"]) < cutoff_len
#         and add_eos_token
#     ):
#         result["input_ids"].append(tokenizer.eos_token_id)
#         result["attention_mask"].append(1)

#     result["labels"] = result["input_ids"].copy()

#     return result
