#!/usr/bin/env python
import datasets
import pandas as pd
import safe as sf

def convert_to_safe(sm, params)

if __name__ == "__main__":
    dt = datasets.load_dataset(
        "/storage/shared_data/manu/zinc_dump/",
        streaming=False,
        num_proc=32,
        cache_dir="/storage/shared_data/manu/.cache",
    )
    dt = dt.cast_column("id", datasets.Value("string"))

    updated_dt = dt.map(update_info, num_proc=32, batched=True, batch_size=5000)
    # unichem_df = pd.read_parquet("/storage/shared_data/manu/unichem", engine="fastparquet")
    # unichem_df["id"] = unichem_df["id"].astype("str")
    # unichem_dt_tmp = datasets.Dataset.from_pandas(unichem_df.drop(columns=["parquet_partition"]))

    # # 80% train, 20% test + validation
    # train_test_valid = unichem_dt_tmp.train_test_split(test_size=0.2)
    # # split the 20 % test into  test and validation
    # test_valid = train_test_valid["test"].train_test_split(test_size=0.5)
    # # gather everyone if you want to have a single DatasetDict
    # unichem_dt = datasets.DatasetDict(
    #     {
    #         "train": train_test_valid["train"],
    #         "test": test_valid["test"],
    #         "validation": test_valid["train"],
    #     }
    # )

    # unichem_dt = unichem_dt.select_columns(["id", "smiles", "source"])
    # print("SAVING the unichem to disk")
    unichem_dt = datasets.load_dataset(
        "/storage/shared_data/manu/processed_unichem",
        cache_dir="/storage/shared_data/manu/.cache",
        num_proc=32,
    )

    test_dt = datasets.concatenate_datasets([unichem_dt["test"], updated_dt["test"]])
    validation_dt = datasets.concatenate_datasets([unichem_dt["validation"], dt["validation"]])
    train_dt = datasets.concatenate_datasets([unichem_dt["train"], dt["train"]])
    dataset = datasets.DatasetDict(dict(train=train_dt, test=test_dt, validation=validation_dt))
    dataset.save_to_disk("/storage/shared_data/manu/processed_zinc_unichem", max_shard_size="1GB")
