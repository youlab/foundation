import json
import os
from datetime import datetime

import numpy as np

from config import (
    DIR_DATA,
    DIR_DATA_PROCESSED,
    SEQ_LEN,
)

from data.config import SPLIT_SCALE, RANDOM_SEED

from data.normalization_functions.utils import generate_train_test_idx, generate_train_test_idx_by_dataset


def analyze_dataset_split(idx_key, train_idx, test_idx, y):
    """
    Analyze which datasets were assigned to train vs test split.
    
    Parameters
    ----------
    idx_key : dict
        Index key from compile_y
    train_idx : np.ndarray
        Indices of training samples
    test_idx : np.ndarray
        Indices of test samples
    y : np.ndarray
        full dataset array
    
    Returns
    -------
    dict
        Dictionary mapping filenames to their split information (n_samples and split assignment)
    """

    dataset_info = {}
    # iterate through all samples and map to source files
    for key in idx_key:
        if not isinstance(key, int):
            continue
        
        sample_idx = key
        sample_data = idx_key[sample_idx]
        filename = None
        for file_key in sample_data:
            if file_key not in ["original_shape", "y.shape"]:
                filename = file_key
                break
        
        if filename is not None:
            if filename not in dataset_info:
                dataset_info[filename] = {
                    "indices": [],
                    "split": None,
                }
            dataset_info[filename]["indices"].append(sample_idx)
    
    # determine whether dataset belongs to train or test
    train_set = set(train_idx)
    for filename in dataset_info:
        indices = dataset_info[filename]["indices"]
        if all(idx in train_set for idx in indices):
            dataset_info[filename]["split"] = "train"
        else: dataset_info[filename]["split"] = "test"
    
    result = {}
    for filename in sorted(dataset_info.keys()):
        n_samples = len(dataset_info[filename]["indices"])
        split = dataset_info[filename]["split"]
        result[filename] = {
            "n_samples": n_samples,
            "split": split,
        }
    
    return result


def interpolate_y(
    y,
    interp_len,
):
    x = np.arange(interp_len).astype(float)
    xp = np.linspace(0, interp_len, y.shape[1]).astype(int).astype(float)
    for i in range(y.shape[0]):
        if i == 0:
            y_temp = np.interp(
                x=x,
                xp=xp,
                fp=y[i, :],
            ).reshape(1, -1)
        else:
            y_temp = np.concatenate(
                (
                    y_temp,
                    np.interp(
                        x=x,
                        xp=xp,
                        fp=y[i, :],
                    ).reshape(1, -1),
                )
            )
    return y_temp


def compile_y(
    seq_len,
    categories,
    interp_len=64,
):
    y_all = None
    idx_key = {}
    for category in categories:
        idx_key[category] = {}
        data_dir = DIR_DATA / category / "processed"

        for _, _, files in os.walk(data_dir):
            break

        for fn in files:
            if fn.find("_y.npy") == -1:
                continue
            y = np.load(data_dir / fn)

            idx_key[category][fn] = {"original_shape": y.shape}
            if np.isnan(y).sum().sum() > 0:
                print(fn)
                continue
                return ValueError
            if y.shape[1] > seq_len:
                for i in range(y.shape[1] // seq_len):
                    if i == 0:
                        y_temp = y[:, :seq_len]
                    else:
                        y_temp = np.concatenate(
                            (
                                y_temp,
                                y[:, i * seq_len : (i + 1) * seq_len],
                            )
                        )
                y_temp = y_temp[y_temp.max(axis=1) != y_temp.min(axis=1), :]
                y = y_temp / y_temp.max(axis=1).reshape(-1, 1)
            elif y.shape[1] < interp_len:
                y = interpolate_y(
                    y=y,
                    interp_len=interp_len,
                )
                if y.shape[1] < seq_len:
                    y = np.concatenate(
                        (
                            y,
                            -1.0 * np.ones([y.shape[0], seq_len - y.shape[1]]),
                        ),
                        axis=1,
                    )
            elif y.shape[1] < seq_len:
                y = np.concatenate(
                    (
                        y,
                        -1.0 * np.ones([y.shape[0], seq_len - y.shape[1]]),
                    ),
                    axis=1,
                )
            idx_key[category][fn]["y.shape"] = y.shape
            if y_all is None:
                y_all = y
                idx_key[category][fn]["y_all_i"] = 0
                for i in range(y.shape[0]):
                    idx_key[i] = {
                        fn: i,
                        "original_shape": idx_key[category][fn]["original_shape"],
                        "y.shape": y.shape,
                    }
            else:
                idx_key[category][fn]["y_all_i"] = y_all.shape[0]
                for i in range(y.shape[0]):
                    idx_key[i + y_all.shape[0]] = {
                        fn: i,
                        "original_shape": idx_key[category][fn]["original_shape"],
                        "y.shape": y.shape,
                    }
                y_all = np.concatenate(
                    (
                        y_all,
                        y,
                    ),
                )
    return y_all, idx_key


def main():
    run_timestamp = datetime.now().isoformat()
    run_dir = DIR_DATA_PROCESSED / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    for categories, label in (
        (["experimental", "simulation",], "all"),
        (["experimental"], "experimental"),
        (["simulation"], "simulation"),
    ):
        print(f"\nWorking on {label}" + 50*"=" )
        y, idx_key = compile_y(
            seq_len=SEQ_LEN,
            categories=categories,
            interp_len=SEQ_LEN,
        )

        print(f"Sequence length {SEQ_LEN} y shape {y.shape}")

        print("Splitting dataset into train and test...")
        print(f"Random seed used for splitting: {RANDOM_SEED}")
        
        if SPLIT_SCALE == "by_dataset":
            train_idx, test_idx = generate_train_test_idx_by_dataset(idx_key=idx_key)
        else:
            train_idx, test_idx = generate_train_test_idx(n=y.shape[0], use_random_seed=False)

        # compute split statistics
        train_shape = tuple(y[train_idx].shape)
        test_shape = tuple(y[test_idx].shape)
        train_pct = len(train_idx) / y.shape[0] * 100
        test_pct = len(test_idx) / y.shape[0] * 100
        dataset_split_info = analyze_dataset_split(idx_key, train_idx, test_idx, y)
        
        # print split statistics
        print(f"Train shape: {train_shape}")
        print(f"Test shape: {test_shape}")
        print(f"Train %: {train_pct:.3f}%")
        print(f"Test %: {test_pct:.3f}%")

        # print("\n" + "="*80)
        # print("DATASET SPLIT ANALYSIS")
        # print("="*80)
        # for filename in sorted(dataset_split_info.keys()):
        #     n_samples = dataset_split_info[filename]["n_samples"]
        #     split = dataset_split_info[filename]["split"]
        #     print(f"{filename:30s} | Samples: {n_samples:6d} | Split: {split}")
        # print("="*80)

        # save npz file
        np.savez(
            run_dir / f"{SEQ_LEN}_{label}_y.npz",
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        # save idx_key json
        with open(
            run_dir / f"{SEQ_LEN}_{label}_idx_key.json",
            "w",
        ) as fp:
            json.dump(idx_key, fp)
        
        # save config
        data_config = {
            "datetime": run_timestamp,
            "random_seed": int(RANDOM_SEED),
            "train_shape": list(train_shape),
            "test_shape": list(test_shape),
            "train_percentage": train_pct,
            "test_percentage": test_pct,
            "dataset_split": dataset_split_info,
        }
        
        with open(
            run_dir / f"data_config_{label}.json",
            "w",
        ) as fp:
            json.dump(data_config, fp, indent=2)
