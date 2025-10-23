import numpy as np
import pandas as pd

from config import PATH_FUJITA_DATA
from data.normalization_functions.utils import get_t_cols


def get_df_x(x_min=0.01):
    df = pd.read_csv(PATH_FUJITA_DATA)

    t_cols, _ = get_t_cols(p=df)
    df["source"] = [a.split("-")[0] for a in df.file]
    df["experiment_id"] = df.source + "_" + df.media + "_" + df.rep

    x = np.zeros_like(df.loc[:, t_cols].to_numpy())
    for replicate in df.rep.unique():
        mask_replicate = df.rep == replicate
        for media in df.loc[mask_replicate].media.unique():
            mask_media = mask_replicate & (df.media == media)
            for source in df.loc[mask_media].source.unique():
                mask_source = mask_media & (df.source == source)
                x_ = df.loc[mask_source, t_cols].to_numpy()
                x[mask_source, :] = x_ / x_.sum(axis=0)

    mask = x.mean(axis=1) > x_min

    df = df.loc[mask, :].copy()
    x = x[mask].copy()
    return df, x
