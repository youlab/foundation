import os

import numpy as np
import pandas as pd
from data.normalization_functions.utils import get_ys, save_vals

from config import DIR_DATA

def normalize_data_rohan():
    data_dir = DIR_DATA / "experimental" / "rohan"

    for _, _, files in os.walk(data_dir):
        break
    
    ts = {}
    y_raws = {}
    
    for fn in files:
        if fn == 'test.csv':
            continue
        if fn == 'August2023-tetA-GFP-clone-growth-curves.csv':
            continue
        if fn[-4:] == ".csv":
            df = pd.read_csv(data_dir / fn)
            dfp = df.pivot(
                columns=["hours"],
                index=["Well", "Tet", "Treatment",],
                values=["RawOD600"],
            )
            y_raws[fn] = dfp.to_numpy()
            ts[fn] = dfp.loc[:, "RawOD600"].columns.to_numpy()

    m = 0
    for key in y_raws.keys():
        m += y_raws[key].shape[0]
    
    y_raw = np.zeros([m, y_raws[key].shape[1]],)
    t = np.zeros([m, ts[key].shape[0]],)
    
    idx = 0
    for key in y_raws.keys():
        m = y_raws[key].shape[0]
        y_raw[idx:idx + m] = y_raws[key]
        t[idx:idx + m] = np.tile(ts[key], (m, 1))
        
        idx += m
    
    y, y_min, y_max = get_ys(
        y_raw=y_raw,
        filename=fn,
    )
    val = "OD600"
    
    save_vals(
        category="experimental",
        label=f"rohan_216_{val.lower()}",
        t=t,
        y_raw=y_raw,
        y=y,
        y_min=y_min,
        y_max=y_max,
    )

    fn = 'August2023-tetA-GFP-clone-growth-curves.csv'
    df = pd.read_csv(data_dir / fn)
    
    dfp = df.pivot(
        columns=["hours"],
        index=["Tet", "Plasmid", "Population", "Date",],
        values=["OD600"],
    )

    y_raw = dfp.to_numpy()
    y, y_min, y_max = get_ys(
        y_raw=y_raw,
        filename=fn,
    )

    t = dfp.loc[:, "OD600"].columns.to_numpy()

    save_vals(
        category="experimental",
        label=f"rohan_180_{val.lower()}",
        t=t,
        y_raw=y_raw,
        y=y,
        y_min=y_min,
        y_max=y_max,
    )
