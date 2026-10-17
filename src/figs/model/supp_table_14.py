import json
import os

import numpy as np
import pandas as pd

from config import (
    DIR_DATA_PROCESSED,
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS_MODEL_NEW_SPLITS,
)


DIR_SPLITS = DIR_DATA_PROCESSED / "dataset_level_splitting_analysis"

# dataset-level splits, in the order shown as Split 1-6 (same order as figs/model/dataset_level_splits.py)
SPLIT_TIMESTAMPS = [
    "2026-05-05T01:11:54.987230",
    "2026-05-05T01:16:08.370504",
    "2026-05-05T01:20:52.269355",
    "2026-05-05T01:25:29.368284",
    "2026-05-05T23:07:34.714356",
    "2026-05-10T19:44:45.339589",
]

N_FILES = 1204
N_CURVES = 376091


def main():
    '''
    Supplementary Table S14: summary of the original curve-level split and the six dataset-level splits used to retrain
    A7X (seeds, file and curve counts, proportions, final test loss, R2), plus the per-file train/test assignments.
    '''
    rows = []

    # original curve-level split
    data_all = np.load(DIR_SPLITS / "old_split" / "128_2024-08-16_all.npz")
    train_idx, test_idx = data_all["train_idx"], data_all["test_idx"]
    n_total = data_all["y"].shape[0]
    assert n_total == N_CURVES, f"original split: {n_total} curves, expected {N_CURVES}"
    assert train_idx.size + test_idx.size == n_total, "original split: train and test do not cover the corpus"
    r2_train, r2_test = np.load(DIR_SPLITS / "old_split" / "old_model" / "r2s.npy")
    rows.append({
        "split": "Original (curve-level)",
        "random_seed": "block-wise",
        "files_train": np.nan,
        "files_test": np.nan,
        "curves_train": int(train_idx.size),
        "curves_test": int(test_idx.size),
        "train_percentage": 100 * train_idx.size / n_total,
        "epochs": np.nan,
        "final_test_loss": np.nan,
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
    })

    # file name -> {"type", "n_curves", "split_k"} across the six dataset-level splits
    assignments = {}

    for k, ts in enumerate(SPLIT_TIMESTAMPS, start=1):
        run_dir = DIR_SPLITS / f"run_{ts}"
        with open(run_dir / "data_config_all.json", "r") as fp:
            config = json.load(fp)
        with open(run_dir / "data_config_simulation.json", "r") as fp:
            sim_files = set(json.load(fp)["dataset_split"].keys())

        dataset_split = config["dataset_split"]
        assert len(dataset_split) == N_FILES, f"Split {k}: {len(dataset_split)} files, expected {N_FILES}"

        # counts[(type, split)] = [n_curves, n_files]
        counts = {(t, s): [0, 0] for t in ("exp", "sim") for s in ("train", "test")}
        for file_name, info in dataset_split.items():
            file_type = "sim" if file_name in sim_files else "exp"
            counts[(file_type, info["split"])][0] += info["n_samples"]
            counts[(file_type, info["split"])][1] += 1

            entry = assignments.setdefault(
                file_name,
                {"type": "simulation" if file_type == "sim" else "experimental", "n_curves": info["n_samples"]},
            )
            assert entry["n_curves"] == info["n_samples"], f"{file_name}: curve count differs across splits"
            entry[f"split_{k}"] = info["split"]

        n_train = counts[("exp", "train")][0] + counts[("sim", "train")][0]
        n_test = counts[("exp", "test")][0] + counts[("sim", "test")][0]
        assert n_train == config["train_shape"][0], f"Split {k}: train curves {n_train} != {config['train_shape'][0]}"
        assert n_test == config["test_shape"][0], f"Split {k}: test curves {n_test} != {config['test_shape'][0]}"
        assert n_train + n_test == N_CURVES, f"Split {k}: {n_train + n_test} curves, expected {N_CURVES}"

        test_loss = np.load(DIR_RESULTS_MODEL_NEW_SPLITS / f"a7x_08_run_{ts}_all_1000epochs" / "test_loss.npy")  # (n_epochs,)
        r2_train, r2_test = np.load(run_dir / f"a7x_08_run_{ts}_all_1000epochs" / "r2s.npy")

        rows.append({
            "split": f"Split {k}",
            "random_seed": config["random_seed"],
            "files_train": counts[("exp", "train")][1] + counts[("sim", "train")][1],
            "files_test": counts[("exp", "test")][1] + counts[("sim", "test")][1],
            "curves_train": n_train,
            "curves_test": n_test,
            "train_percentage": config["train_percentage"],
            "epochs": test_loss.size,
            "final_test_loss": float(test_loss[-1]),
            "r2_train": float(r2_train),
            "r2_test": float(r2_test),
        })

    table = pd.DataFrame(rows)

    df_assignments = pd.DataFrame.from_dict(assignments, orient="index")
    df_assignments.index.name = "file"
    df_assignments = df_assignments.sort_index()
    assert len(df_assignments) == N_FILES, f"{len(df_assignments)} files in assignments, expected {N_FILES}"
    assert not df_assignments.isna().any().any(), "some files are missing an assignment in at least one split"

    os.makedirs(DIR_FIGS_MANUSCRIPT, exist_ok=True)
    table.to_csv(DIR_FIGS_MANUSCRIPT / "supp_table_14.csv", index=False)
    df_assignments.to_csv(DIR_FIGS_MANUSCRIPT / "supp_table_14_file_assignments.csv")

    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(table)
    print(f"Saved table to {DIR_FIGS_MANUSCRIPT / 'supp_table_14.csv'}")
    print(f"Saved file assignments to {DIR_FIGS_MANUSCRIPT / 'supp_table_14_file_assignments.csv'}")
