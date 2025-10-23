import json

import numpy as np

from applications.antibiotics.carolyn.data import get_data
from applications.antibiotics.config import TRAIN_SIZES
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
from config import DIR_CACHE_CAROLYN


def noise(
    spread,
):
    return spread * np.random.rand(128) - spread / 2


def main(
    model_type,
    z_dim,
    cross_val=-1,
):
    # AUGMENT_DATA is a parameter to allow for augmenting the data by applying some artifical 
    # noise to the dataset. I hypothesize that this datasets counts as "sparse", such that there
    # is not enough data to generate meaningful raw predictions, and that's why we see latent and
    # raw so close with less training curves.
    if cross_val not in {0, 1, 2, 3, 4,}:
        raise ValueError(f"Cross validation should be 0 - 4 but is {cross_val}")

    AUGMENT_DATA = False
    df = get_data()

    x_raw = interpolate_y(
        y=df.loc[:, 0:98].to_numpy(),
    )
    if AUGMENT_DATA:
        augmentations = [x_raw]
        n_augmentations = 30
        for i in range(n_augmentations):
            augmentations.append(
                (np.random.rand() * 0.1 + 0.95) * x_raw + noise(spread=0.05,)
            )
        x_raw = np.concatenate(augmentations)

    x_latents, x_latents_fine_tuned, x_latents_end2end = None, None, None
    for antibiotic in [
        "SAM",
        "GM",
        "SXT",
        "CIP",
    ]:
        tgt = df.loc[:, antibiotic].to_numpy()
        if AUGMENT_DATA:
            tgt = np.tile(tgt, n_augmentations + 1)

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
            )

        with open(
            DIR_CACHE_CAROLYN
            / f"classify_antibiotics_{antibiotic}_{cross_val}_{model_type}_{z_dim}.json",
            "w",
        ) as fp:
            json.dump(results, fp)
