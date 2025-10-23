import os

from huggingface_hub import HfApi

from config import DIR_DATA


def main():
    api = HfApi()
    for _, dirs, _ in os.walk(
        DIR_DATA
        / "experimental"
    ):
        break
    for directory in dirs:
        print(f"Uploading {directory}")
        api.upload_folder(
            folder_path=DIR_DATA / "experimental" / directory,
            repo_id="you-lab/foundation-model-data-experimental",
            path_in_repo=directory,
            repo_type="dataset",
            allow_patterns="*.csv",
            delete_patterns="*.csv",
        )

    api.upload_folder(
        folder_path=DIR_DATA / "experimental" / "processed",
        repo_id="you-lab/foundation-model-data-experimental",
        path_in_repo="processed",
        repo_type="dataset",
        allow_patterns="*.npy",
        delete_patterns="*.npy",
    )
