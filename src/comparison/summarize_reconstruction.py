import json
import os

import numpy as np
import pandas as pd

from comparison.config import (
    NOW_TEXT,
    get_params,
)
from config import DIR_RESULTS_MODEL_COMPARISON


def get_task_id(fn):
    if fn.find("task_id") > -1:
        return int(fn.split("task_id_")[-1].replace(".pth", ""))
    return -1


def process_model_name(val):
    if (
        val.find("2025-01-28") > -1
    ) or (
        val.find("2025-01-30") > -1
    ) or (
        val.find("2025-02-21") > -1
    ) or (
        val.find("2025-02-27") > -1
    ):
        return val.split("_")[1]
    return "conli"


def process_z_dim(val):
    if (
        val.find("2025-01-28") > -1
    ) or (
        val.find("2025-01-30") > -1
    ) or (
        val.find("2025-02-21") > -1
    ) or (
        val.find("2025-02-27") > -1
    ):
        return int(val.split("_")[2])
    return 8


def add_transformer_params(df):
    params = get_params()
    df["task_id"] = [get_task_id(fn=model_name) for model_name in df.model_name]
    for p, i in [("lr", 0,), ("n_layers", 2,), ("random_masking", 3,), ("loss_fcn", 4,),]:
        df[p] = 0
        df.loc[df.task_id > -1, p] = [params[task_id][i] for task_id in df.loc[df.task_id > -1, "task_id"]]
    
    for p, i in [("z_dim", 0,), ("n_heads", 1,)]:
        df[p] = 0
        df.loc[df.task_id > -1, p] = [params[task_id][1][i] for task_id in df.loc[df.task_id > -1, "task_id"]]
    return df


def add_model_type_and_z_dim(df):
    NEW_NAMES = {
        "conli": "A7X",
        "livae": "MNM",
        "vae": "VB",
        "pca": "PR",
        "transformer": "MCR",
    }

    mask = df.task_id == -1
    df.loc[mask, "model_type"] = [process_model_name(val) for val in df.loc[mask, "model_name"]]
    df.loc[mask, "model_type"] = [NEW_NAMES[model_type] for model_type in df.loc[mask, "model_type"]]
    df.loc[mask, "z_dim"] = [process_z_dim(val) for val in df.loc[mask, "model_name"]]
    df.loc[~mask, "model_type"] = "MCR"
    return df


def get_best_ones(
    df,
    model_type,
):
    mask = df.model_type == model_type
    best_ones = []
    for z_dim in df.loc[mask, "z_dim"].unique():
        best_ones.append(
            df.loc[mask & (df.z_dim == z_dim)].sort_values(
                "r2_test_all",
                ascending=False,
            ).iloc[0].model_name
        )
    return best_ones


def filter_best_models(df):
    for model_type in df.model_type.unique():
        best_ones = get_best_ones(
            df=df,
            model_type=model_type,
        )
        df = df.loc[np.array([model_name in best_ones for model_name in df.model_name]) | (df.model_type != model_type)]
    return df


def get_file_names(dir_cache):
    for _, _, files in os.walk(dir_cache):
        break
    
    good_files = []
    for file_name in files:
        if file_name.find(NOW_TEXT) > -1:
            if file_name.find("recon") > -1:
                good_files.append(file_name)
    return good_files


def main():
    COLUMNS_TO_KEEP = [
        'model_name',
        'model_type',
        'z_dim',
        'r2_train_all',
        'r2_test_all',
        'r2_train_sim',
        'r2_test_sim',
        'r2_train_exp',
        'r2_test_exp',
    ]

    FILTER_FOR_BEST_MODELS = True

    df = pd.concat(
        [
            pd.read_csv(
                DIR_RESULTS_MODEL_COMPARISON
                / file_name,
                index_col=0,
            ) for file_name in get_file_names(dir_cache=DIR_RESULTS_MODEL_COMPARISON)
        ],
    )

    df = add_transformer_params(df)
    df = add_model_type_and_z_dim(df)
    if FILTER_FOR_BEST_MODELS:
        df = filter_best_models(df=df)

    df = df.sort_values(
        [
            "model_type",
            "z_dim",
        ],
    ).loc[:, COLUMNS_TO_KEEP]

    df.to_csv(
        DIR_RESULTS_MODEL_COMPARISON
        / "reconstruction.csv",
    )
    best_models = {}
    for _, row in df.iterrows():
        best_models[row.model_name] = (
            row.model_type,
            row.z_dim,
        )

    with open(
        DIR_RESULTS_MODEL_COMPARISON
        / "best_models.json",
        "w",
    ) as fp:
        json.dump(
            obj=best_models,
            fp=fp,
        )

    return df
