import os
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import load_dataset


def save_stream_to_parquet(dataset_name, num_records, output_dir="data/", name=None):
    if name:
        dataset_stream = load_dataset(dataset_name, split="train", streaming=True, name=name)
    else:
        dataset_stream = load_dataset(dataset_name, split="train", streaming=True)

    dataset_name = dataset_name.replace("/", "_").replace(" ", "_")
    output_file = os.path.join(output_dir, f"{dataset_name}_{num_records}_records.parquet")

    if os.path.exists(output_file):
        print(f"File '{output_file}' already exists. Skipping download.")
        return output_file

    os.makedirs(output_dir, exist_ok=True)

    records = []
    for i, record in enumerate(dataset_stream):
        records.append(record)
        if i + 1 >= num_records:
            break

    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_file)
    print(f"Saved {num_records} records from '{dataset_name}' to {output_file}")
    return output_file
