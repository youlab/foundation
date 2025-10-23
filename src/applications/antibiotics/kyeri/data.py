
import pandas as pd

from applications.antibiotics.kyeri.config import (
    keys,
    processes,
)
from applications.utils.data import (
    get_t_cols,
    interpolate_y,
)
from config import DIR_DATA_KYERI


def get_df():
    dfs = []
    for i in range(3):
        for _, fn in enumerate(keys[i].keys()):
            df = pd.read_csv(DIR_DATA_KYERI / fn, index_col=0,).dropna()
            df = processes[fn](df=df)
            t_cols, _ = get_t_cols(p=df)
            y = interpolate_y(y=df.loc[:, t_cols].to_numpy())
            non_t_cols = []
            for col in df.columns:
                if col not in t_cols:
                    non_t_cols.append(col)
            
            dfs.append(pd.concat(
                (
                    df.loc[:, non_t_cols].reset_index(drop=True),
                    pd.DataFrame(y).reset_index(drop=True),
                    pd.DataFrame([fn] * df.shape[0], columns=["file_name"],),
                ),
                axis=1,
            ))
    df = pd.concat(dfs)
    good_cols = []
    for col in df.columns:
        if (
            isinstance(col, int)
        ) or (
            (
                col.find("before_ab") == -1
            ) and (
                col != "well"
            ) and (
                col != 'Cycle Nr.'
            )
        ):
            good_cols.append(col)
    return df.loc[:, good_cols].copy()
