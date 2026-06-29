import json
import os

import numpy as np

from applications.antibiotics.kyeri.data import get_df
from applications.antibiotics.generate_results import main as generate_results
from applications.config import (
    EPOCHS_FINE_TUNED,
    LR_FINE_TUNED,
    EPOCHS_END2END,
    LR_END2END,
)
from config import (
    DIR_CACHE_KYERI,
    SEQ_LEN,
)
from applications.antibiotics.config import TRAIN_SIZES


def main(
    model_type,
    z_dim,
    cross_val=-1,
):
    df = get_df()

    mask = (df.loc[:, "antibiotic_conc"] > 0)
    mask = mask & (df.loc[:, "Reading"] == "OD600")
    data = df.loc[mask].copy()
    x_raw = data.loc[:, np.arange(SEQ_LEN)].to_numpy()
    tgt = data.loc[:, "antibiotic_type"].to_numpy()
    results = generate_results(
        x_raw=x_raw,
        tgt=tgt,
        model_type=model_type,
        z_dim=z_dim,
        epochs_fine_tuned=EPOCHS_FINE_TUNED,
        lr_fine_tuned=LR_FINE_TUNED,
        epochs_end2end=EPOCHS_END2END,
        lr_end2end=LR_END2END,
        classify=True,
        detailed_binary=False,
        train_sizes=TRAIN_SIZES,
        cross_val=cross_val,
    )

    os.makedirs(DIR_CACHE_KYERI, exist_ok=True)
    with open(
        DIR_CACHE_KYERI
        / f"classify_antibiotics_{cross_val}_{model_type}_{z_dim}.json",
        "w",
    ) as fp:
        json.dump(results, fp)
