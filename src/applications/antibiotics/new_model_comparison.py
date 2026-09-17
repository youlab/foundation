import json
import os

import numpy as np
import pandas as pd

from applications.antibiotics.carolyn.data import get_data
from applications.antibiotics.generate_results import (
    main as generate_results,
    split_data_and_get_results,
)
from applications.config import (
    EPOCHS_FINE_TUNED,
    LR_FINE_TUNED,
    EPOCHS_END2END,
    LR_END2END,
)
from applications.utils.data import interpolate_y
from config import (
    DIR_CACHE_CAROLYN,
    DIR_RESULTS_ANTIBIOTICS,
    SEQ_LEN,
)

# The five pretrained architectures compared in supplemental figure S8
MODEL_TYPES = [
    "A7X",
    "MCR",
    "MNM",
    "PR",
    "VB",
]

# SAM must come first, since the first antibiotic loads the model and computes the latents and the
# remaining reuse them. this matches applications/antibiotics/carolyn/main.py.
ANTIBIOTICS = [
    "SAM",
    "GM",
    "SXT",
    "CIP",
]

# 0.8 is the full training fold since loop_for_results short circuits this value, so no nested
# subsampling happens and the results are identical to the 100% point of the 9-size sweep in
# applications/antibiotics/config.py.
TRAIN_SIZES = [0.8]


def cache_path(
    antibiotic,
    cross_val,
    model_type,
    z_dim,
):
    """Path of the per-fold result cache, kept separate from the original carolyn caches."""
    return (
        DIR_CACHE_CAROLYN
        / f"new_model_comparison_{antibiotic}_{cross_val}_{model_type}_{z_dim}.json"
    )


def run_model(
    model_type,
    z_dim,
    cross_val,
):
    """Run the resistance classification for one architecture and one cross validation fold."""
    if cross_val not in {0, 1, 2, 3, 4,}:
        raise ValueError(f"Cross validation should be 0 - 4 but is {cross_val}")

    df = get_data()
    x_raw = interpolate_y(
        y=df.loc[:, 0:98].to_numpy(),
    )
    # x_raw has shape (n_curves, SEQ_LEN)
    assert x_raw.shape[1] == SEQ_LEN, f"x_raw should be (n, {SEQ_LEN}) but is {x_raw.shape}"

    x_latents, x_latents_fine_tuned, x_latents_end2end = None, None, None
    for antibiotic in ANTIBIOTICS:
        tgt = df.loc[:, antibiotic].to_numpy()

        if x_latents is None:
            (
                results,
                x_latents,
                x_latents_fine_tuned,
                x_latents_end2end,
            ) = generate_results(
                x_raw=x_raw,
                tgt=tgt,
                model_type=model_type,
                z_dim=z_dim,
                classify=True,
                detailed_binary=True,
                epochs_fine_tuned=EPOCHS_FINE_TUNED,
                lr_fine_tuned=LR_FINE_TUNED,
                epochs_end2end=EPOCHS_END2END,
                lr_end2end=LR_END2END,
                train_sizes=TRAIN_SIZES,
                return_latent_vectors=True,
                cross_val=cross_val,
            )
            # x_latents has shape (n_curves, z_dim + 1), last column is the appended max
            assert x_latents.shape == (
                x_raw.shape[0],
                z_dim + 1,
            ), f"x_latents should be ({x_raw.shape[0]}, {z_dim + 1}) but is {x_latents.shape}"
        else:
            results = split_data_and_get_results(
                x_raw=x_raw,
                x_latents=x_latents,
                x_latents_fine_tuned=x_latents_fine_tuned,
                x_latents_end2end=x_latents_end2end,
                tgt=tgt,
                classify=True,
                detailed_binary=True,
                train_sizes=TRAIN_SIZES,
                cross_val=cross_val,
            )

        os.makedirs(DIR_CACHE_CAROLYN, exist_ok=True)
        with open(
            cache_path(
                antibiotic=antibiotic,
                cross_val=cross_val,
                model_type=model_type,
                z_dim=z_dim,
            ),
            "w",
        ) as fp:
            json.dump(results, fp)


def read_f1(
    antibiotic,
    cross_val,
    model_type,
    z_dim,
):
    """Read the raw and latent F1 scores of one fold at the full training size."""
    with open(
        cache_path(
            antibiotic=antibiotic,
            cross_val=cross_val,
            model_type=model_type,
            z_dim=z_dim,
        ),
        "r",
    ) as fp:
        res = json.load(fp)

    # only one training size was run
    train_sizes = [key for key in res.keys() if key != "test_size"]
    if len(train_sizes) != 1:
        raise ValueError(f"Expected one training size but found {train_sizes}")

    return (
        res[train_sizes[0]]["raw_accuracy"]["f1_score"],
        res[train_sizes[0]]["latent_accuracy"]["f1_score"],
    )


def collect_f1(
    z_dim,
):
    """Aggregate the per-fold F1 scores into one tidy table and write it to results."""
    records = []
    for antibiotic in ANTIBIOTICS:
        raw_means = []
        raw_stds = []
        for model_type in MODEL_TYPES:
            raw = []
            latent = []
            for cross_val in range(5):
                raw_f1, latent_f1 = read_f1(
                    antibiotic=antibiotic,
                    cross_val=cross_val,
                    model_type=model_type,
                    z_dim=z_dim,
                )
                raw.append(raw_f1)
                latent.append(latent_f1)

            raw_means.append(np.mean(raw))
            raw_stds.append(np.std(raw, ddof=1))
            records.append(
                {
                    "antibiotic": antibiotic,
                    "model": model_type,
                    "f1_mean": np.mean(latent),
                    "f1_std": np.std(latent, ddof=1),
                    "n_folds": len(latent),
                }
            )

        assert np.allclose(
            raw_means,
            raw_means[0],
        ), f"raw F1 for {antibiotic} differs across models: {dict(zip(MODEL_TYPES, raw_means))}"

        records.append(
            {
                "antibiotic": antibiotic,
                "model": "raw",
                "f1_mean": raw_means[0],
                "f1_std": raw_stds[0],
                "n_folds": 5,
            }
        )

    df = pd.DataFrame.from_records(records)
    os.makedirs(DIR_RESULTS_ANTIBIOTICS, exist_ok=True)
    out_path = DIR_RESULTS_ANTIBIOTICS / f"new_model_comparison_f1_{z_dim}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {df.shape[0]} rows to {out_path}")
    return df


def main(
    z_dim=8,
):
    """Rerun the resistance classification for every architecture at one latent dimension."""
    for model_type in MODEL_TYPES:
        for cross_val in range(5):
            done = all(
                cache_path(
                    antibiotic=antibiotic,
                    cross_val=cross_val,
                    model_type=model_type,
                    z_dim=z_dim,
                ).exists()
                for antibiotic in ANTIBIOTICS
            )
            if done:
                print(f"Skipping {model_type} z_dim {z_dim} fold {cross_val}, cache exists")
                continue

            print(f"Running {model_type} z_dim {z_dim} fold {cross_val}")
            run_model(
                model_type=model_type,
                z_dim=z_dim,
                cross_val=cross_val,
            )

    return collect_f1(z_dim=z_dim)
