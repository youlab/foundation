import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from applications.utils import (
    get_t_cols,
    get_t_idx,
    interpolate_y,
    plot_regression,
    calc_latents,
)
from config import (
    DIR_DATA_HELENA,
    SEQ_LEN,
)
from data.normalization_functions.utils import generate_train_test_idx


def get_inputs(
    model,
    normalize_concentration=True,
    log_concentration=True,
    return_plasmids=False,
):

    
    df = pd.read_csv(
        DIR_DATA_HELENA
        / "helena_data.csv",
    )

    df["time_index"] = get_t_idx(df.Hours)

    df = df.pivot(
        index=["strain_background", "plasmid_name", "mixed_flag", "antibiotic_name", "inhibitor_name", "[A]", "[I]", "Replicate",],
        columns=["time_index",],
        values=["Hours", "OD600", "GFP", "BFP",],
    )

    df = df.reset_index()
    df = df.drop(columns=["Hours"], level=0,)

    x_out = np.zeros((df.shape[0], 27))
    y_out = np.zeros((df.shape[0], 128 * 3))
    for i, val in enumerate(["OD600", "GFP", "BFP",]):
        t_cols, t = get_t_cols(p=df.loc[:, val])
        mask = ~pd.isnull(df.loc[:, val]).any(axis=1)
        y = df.loc[mask, val].loc[:, t_cols].to_numpy()
        y_ = interpolate_y(y=y)
        y_max = y_.max(axis=1).reshape(-1, 1)
        y_min = y_.min(axis=1).reshape(-1, 1)
        y_norm = (y_ - y_min) / (y_max - y_min)

        latents = calc_latents(
            x=y_norm.reshape(-1, 1, SEQ_LEN)
        )
        
        x_out[mask, i * 9:i*9+8] = latents.cpu().detach().numpy()
        x_out[mask, i * 9 + 8] = y_max.flatten()

        y_out[mask, i * 128:(i + 1) * 128] = (y_ - y_.min()) / (y_.max() - y_.min())
    
    encoder = OrdinalEncoder()
    y_class = encoder.fit_transform(df.inhibitor_name.to_numpy().reshape(-1, 1)).flatten()
    if return_plasmids:
        encoder_plasmids = OrdinalEncoder()
        y_plasmids = encoder_plasmids.fit_transform(df.plasmid_name.to_numpy().reshape(-1, 1)).flatten()
    
    y_conc = df.loc[:, "[I]"].to_numpy()
    if normalize_concentration:
        for c in np.unique(y_class):
            if c != "none":
                mask = y_class == c
                max_conc = y_conc[mask].max()
                if max_conc != 0:
                    y_conc[mask] = y_conc[mask] / max_conc
    
    if log_concentration:
        y_conc[y_conc == 0] = y_conc[y_conc != 0].min() / 10
        y_conc = np.log10(y_conc)

    if return_plasmids:
        return x_out, y_out, y_class, y_conc, encoder, y_plasmids, encoder_plasmids
    return x_out, y_out, y_class, y_conc, encoder


def get_train_and_test_sets(
    x,
    y_class,
    y_conc,
    train_ratio,
):
    train_idx, test_idx = generate_train_test_idx(
        n=x.shape[0],
        train_ratio=train_ratio,
    )
    
    x_train = x[train_idx, :]
    x_test = x[test_idx, :]
    
    y_class_train = y_class[train_idx]
    y_class_test = y_class[test_idx]
    
    y_conc_train = y_conc[train_idx]
    y_conc_test = y_conc[test_idx]

    return x_train, x_test, y_class_train, y_class_test, y_conc_train, y_conc_test
