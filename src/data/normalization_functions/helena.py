import pandas as pd

from config import DIR_DATA
from data.normalization_functions.utils import (
    get_ys,
    save_vals,
)


def normalize_data_helena():
    
    def get_t_idx(column: pd.Series,):
        assert isinstance(column, pd.Series), f"'column' should be pd.Series but is {type(column)}"
        greater_than_next = column.to_numpy()[1:] > column.to_numpy()[:-1]
        t_idx = [0]
        for val in greater_than_next:
            if val:
                t_idx.append(t_idx[-1] + 1)
            else:
                t_idx.append(0)

        return t_idx
    
    FILENAME = "helena_data.csv"
    df = pd.read_csv(
        DIR_DATA
        / "experimental"
        / "helena"
        / FILENAME
    )
    
    df["time_index"] = get_t_idx(df.Hours)

    df = df.pivot(
        index=["strain_background", "plasmid_name", "mixed_flag", "antibiotic_name", "inhibitor_name", "[A]", "[I]", "Replicate",],
        columns=["time_index",],
        values=["Hours", "OD600", "GFP", "BFP",],
    )

    for val in ["OD600", "GFP",]:
        y_raw = df.loc[:, val].to_numpy()
        y, y_min, y_max = get_ys(
            y_raw=y_raw,
            filename=FILENAME,
        )
        t = df.Hours.to_numpy()

        save_vals(
            category="experimental",
            label=f"helena_{val.lower()}",
            t=t,
            y_raw=y_raw,
            y=y,
            y_min=y_min,
            y_max=y_max,
        )
