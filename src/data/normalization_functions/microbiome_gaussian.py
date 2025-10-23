import os

import numpy as np
import pandas as pd
from data.normalization_functions.utils import (
    get_ys,
    save_vals,
    get_t_cols,
)

from config import DIR_DATA

TAXA_LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species",]


def add_taxa_columns(df):
    for level in TAXA_LEVELS:
        df[level] = ""

    for idx in df.index:
        idx_str = idx.split("\t")[2].split(";")
        if len(idx_str) != 7:
            raise ValueError(idx)
        df.loc[idx, "kingdom"] = idx_str[0].split("__")[1]
        df.loc[idx, "phylum"] = idx_str[1].split("__")[1]
        df.loc[idx, "class"] = idx_str[2].split("__")[1]
        df.loc[idx, "order"] = idx_str[3].split("__")[1]
        df.loc[idx, "family"] = idx_str[4].split("__")[1]
        df.loc[idx, "genus"] = idx_str[5].split("__")[1]
        df.loc[idx, "species"] = idx_str[6].split("__")[1].split("\n")[0]
    return df


def fill_missing_taxa(df):
    for i in range(1, 7):
        mask = df.loc[:, TAXA_LEVELS[i]] == ''
        df.loc[mask, TAXA_LEVELS[i]] = df.loc[mask, TAXA_LEVELS[i-1]]
    return df


def compile_y(df):
    """
    This function compiles the microbiome data at multiple taxonomic levels, ranging from phylum down to species
    """
    y_all = None
    for level in TAXA_LEVELS[1:]:
        y = df.groupby(level).agg('sum', numeric_only=True,).to_numpy()
        y = y[y.max(axis=1) > 0]
        y_star = y / y.sum(axis=0)
        if y_all is None:
            y_all = y_star
        else:
            y_all = np.concatenate((y_all, y_star))
    return y_all


def file_exists(file_path):
    try:
        np.load(file_path)
        return True
    except: 
        return False


def normalize_data_microbiome(
    name,
    num_days,
    length_input,
    overlap,
    val="OD600",
):
    assert isinstance(name, str), TypeError(f"'name' should be string but is {type(name)}")

    y_raws = {}
    ts = {}

    data_dir = DIR_DATA / "experimental" / name

    for _, _, files in os.walk(data_dir):
        break

    csvs = []
    for fn in files:
        if fn[-4:] == ".csv":
            csvs.append(fn)

    for filename in csvs:
        if file_exists(
            file_path=DIR_DATA
            / "experimental"
            / "processed"
            / f"{name}_{filename[:-4]}_y.npy",
        ):
            print(f"{name}, {filename} already has numpy file.")
            continue
        
        df = pd.read_csv(
            data_dir 
            / filename,
            index_col=0 if name in {"zach", "huge_carin"} else None,
        )
        if "blank" in df.columns:
            df = df.loc[~df.blank, :].copy()
        if "Strain" in df.columns:
            df = df.loc[df.Strain != "media_blank", :].copy()
        if "Sensor" in df.columns:
            df = df.loc[df.Sensor != "blank"].copy()
        if "Strain #" in df.columns:
            df = df.loc[df.loc[:, "Strain #"] != "blank"].copy()
        if "dilution" in df.columns:
            df = df.loc[df.dilution != "con"].copy()
        t_cols, t = get_t_cols(p=df,)
        if name == "huge_carin":
            df = add_taxa_columns(df=df)
            df = fill_missing_taxa(df=df)
            y_raw = compile_y(df=df)
            
        else:
            y_raw = df.loc[:, t_cols].to_numpy()
            y_raw = y_raw / y_raw.sum(axis=0)

        if y_raw.size == 0:
            continue
        y_raw = y_raw[y_raw.max(axis=1) > 0]
        if y_raw.size == 0:
            continue
        y, y_min, y_max = get_ys(
            y_raw=y_raw,
            data_type="microbiome",
            num_days=num_days,
            length_input=length_input,
            overlap=overlap,
        )

        print(f"{name}_{filename[:-4]}", y.shape)

        save_vals(
            category="experimental",
            label=f"{name}_{filename[:-4]}",
            t=np.tile(t, (y_raw.shape[0], 1)),
            y_raw=y_raw,
            y=y,
            y_min=y_min,
            y_max=y_max,
        )
