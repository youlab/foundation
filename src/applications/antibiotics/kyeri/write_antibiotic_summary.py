import json

import numpy as np
import pandas as pd

from config import (
    DIR_CACHE_KYERI,
    DIR_RESULTS_ANTIBIOTICS,
)


def summarize_data(
    model_type,
    z_dim,
    method,
):
    key = {
        1: 1.25,
        12: 12.5,
        25: 25.,
        37: 37.5,
        50: 50.,
        62: 62.5,
        75: 75.,
        87: 87.5,
        100: 100.,
    }

    df = None
    for cross_val in range(5):
        with open(
            DIR_CACHE_KYERI
            / f"{method}_antibiotics_{cross_val}_{model_type}_{z_dim}.json",
            "r",
        ) as fp:
            res = json.load(fp)
        results = []
        for train_size in res.keys():
            if train_size == "test_size":
                continue
            results.append(
                {
                    "train_size": int(train_size),
                    "cross_val": cross_val,
                    "raw_accuracy": res[train_size]["raw_accuracy"],
                    "latent_accuracy": res[train_size]["latent_accuracy"],
                }
            )
        df = pd.concat(
            (
                df,
                pd.DataFrame.from_records(results),
            ),
        )
    df["train_size_pct"] = np.around((df.train_size / df.train_size.max()) * 100).astype(int)
    df["train_size_pct"] = [key[val] for val in df.train_size_pct]

    # df = df.groupby("train_size_pct").agg("mean").reset_index()
    df = (
        df.groupby("train_size_pct")
        .agg(
            train_size=("train_size", "mean"),
            cross_val=("cross_val", "mean"),
            raw_accuracy=("raw_accuracy", "mean"),
            latent_accuracy=("latent_accuracy", "mean"),
            raw_accuracy_std=("raw_accuracy", "std"),
            latent_accuracy_std=("latent_accuracy", "std"),
        )
        .reset_index()
    )


    df.to_csv(
        DIR_RESULTS_ANTIBIOTICS
        / f"{method}_antibiotic_summary_{model_type}_{z_dim}.csv",
    )


def main(
    model_type,
    z_dim,
):
    for method in ["classify", "regress",]:
        summarize_data(
            model_type=model_type,
            z_dim=z_dim,
            method=method,
        )
