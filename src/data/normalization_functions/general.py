import os

import numpy as np
import pandas as pd
from data.normalization_functions.utils import (
    get_ys,
    save_vals,
    get_t_cols,
)

from config import DIR_DATA

excluded_indices = {
    "growth_curves_2023-03-18_3.csv": [
        84, 85, 86, 87, 88, 89,
    ],
    "growth_curves_2023-03-29_1.csv": [
        90, 91, 92, 93, 94, 95,
    ],
    "growth_curves_2023-03-27_1.csv": [
        90, 91, 92, 93, 94, 95,
    ],
    "growth_curves_2023-03-27_3.csv": [
        63,
    ],
    "growth_curves_2023-04-01_1.csv": [
        90, 91, 92, 93, 94, 95,
    ],
    "growth_curves_2023-04-03_1.csv": [
        90, 91, 92, 93, 94, 95,
    ],
    "growth_curves_2023-04-01_2.csv": [
        0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41,
        48, 49, 50, 51, 52, 53, 60, 61, 62, 63, 64, 65, 72, 73, 74, 75, 76, 77, 84, 85, 86, 87, 
        88, 89,
    ]
}


def is_string(val):
    return val == "OVER"


def normalize_data_general(
    name,
    val="OD600",
):
    vectorized_is_string = np.vectorize(is_string)
    assert isinstance(name, str), TypeError(f"'name' should be string but is {type(name)}")

    data_dir = DIR_DATA / "experimental" / name

    for _, _, files in os.walk(data_dir):
        break

    csvs = []
    for fn in files:
        if fn[-4:] == ".csv":
            csvs.append(fn)

    for filename in csvs:
        df = pd.read_csv(
            data_dir 
            / filename,
            index_col=0 if name == "zach" else None,
        )
        if "blank" in df.columns:
            df = df.loc[~df.blank, :].copy()
        if "Strain" in df.columns:
            df = df.loc[df.Strain != "media_blank", :].copy()
        if filename in excluded_indices:
            try:
                df.drop(index=excluded_indices[filename], inplace=True,)
            except Exception as e:
                print(filename, e)
                return KeyError()
        if "Sensor" in df.columns:
            df = df.loc[df.Sensor != "blank"].copy()
        if "Strain #" in df.columns:
            df = df.loc[df.loc[:, "Strain #"] != "blank"].copy()
        if "dilution" in df.columns:
            df = df.loc[df.dilution != "con"].copy()
        t_cols, t = get_t_cols(p=df,)
        y_raw = df.loc[:, t_cols].to_numpy()
        if y_raw.shape[1] < 10:
            continue

        y_raw = y_raw[~vectorized_is_string(y_raw).any(axis=1), :].astype(float)
        y_raw = y_raw[y_raw.max(axis=1,) > 0, :]
        
        y, y_min, y_max = get_ys(
            y_raw=y_raw,
            filename=filename,
        )

        save_vals(
            category="experimental",
            label=f"{name}_{filename[:-4]}",
            t=np.tile(t, (y_raw.shape[0], 1)),
            y_raw=y_raw,
            y=y,
            y_min=y_min,
            y_max=y_max,
        )
