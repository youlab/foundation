import json
import os
from datetime import datetime

import numpy as np

from config import (
    DIR_DATA,
    DIR_DATA_PROCESSED,
    SEQ_LEN,
)
from data.normalization_functions.utils import generate_train_test_idx


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
    today_date = datetime.now().date().isoformat()
    for categories, label in (
        (["experimental", "simulation",], "all"),
        (["experimental"], "experimental"),
        (["simulation"], "simulation"),
    ):
        print(f"Working on {label}")
        y, idx_key = compile_y(
            seq_len=SEQ_LEN,
            categories=categories,
            interp_len=SEQ_LEN,
        )

        print(f"Sequence length {SEQ_LEN} y shape {y.shape}")
        train_idx, test_idx = generate_train_test_idx(n=y.shape[0])

        np.savez(
            DIR_DATA_PROCESSED
            / f"{SEQ_LEN}_{today_date}_{label}_y.npz",
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        with open(
            DIR_DATA_PROCESSED
            / f"{SEQ_LEN}_{today_date}_{label}_idx_key.json",
            "w",
        ) as fp:
            json.dump(
                idx_key,
                fp,
            )
