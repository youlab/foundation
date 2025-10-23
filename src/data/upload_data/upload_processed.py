import os

from huggingface_hub import HfApi

from config import DIR_DATA


def main():
    api = HfApi()
    for file_name in [
        "128_2024-08-16_all_idx_key.json",
        "128_2024-08-16_all.npz",
        "128_2024-08-16_experimental_idx_key.json",
        "128_2024-08-16_experimental.npz",
        "128_2024-08-16_simulation_idx_key.json",
        "128_2024-08-16_simulation.npz",
    ]:
        print(f"Uploading {file_name}")
        api.upload_file(
            path_or_fileobj=DIR_DATA / "processed" / file_name,
            repo_id="you-lab/foundation-model-data-processed",
            path_in_repo=file_name,
            repo_type="dataset",
        )
