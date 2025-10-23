import logging

import numpy as np
import pandas as pd

from config import DIR_RESULTS_CONSORTIA
from applications.consortia.config import get_consortia_for_intrinsic_dims

logger = logging.getLogger(__name__)

def plot_consortia_estimated_intrinsic_dimensions(
    ax,
    s1=10,
    s2=100,
    fs_label=16,
    fs_ticks=14,
    fs_text=16,
):
    consortia = get_consortia_for_intrinsic_dims()
    df = pd.read_csv(
        DIR_RESULTS_CONSORTIA
        / "dims.csv",
        index_col=0,
    )
    df = df[~(df.file_path.str.find("binary") > -1)].copy()
    df = df[~(df.file_path.str.find("oscillation") > -1)].copy()
    ax.grid(visible=True, which='both', axis='y',)
    
    ax.scatter(
        np.arange(df.shape[0]),
        df.intrinsic_dimensions.to_numpy(),
        s=s1,
        color="cornflowerblue",
    )
    
    for i, consortium in enumerate(consortia):
        logger.info(f"Working on {consortium}")
        mask = (f"/{consortium[21:25]}/{consortium[26:28]}/{consortium[29:]}.txt" == df.file_path).to_numpy()
        if mask.sum() == 0:
            pass
        else:
            logger.warning(f"Mask sum = 0")
        x, y = np.where(mask)[0][0], float(df.loc[mask, "intrinsic_dimensions"])
        ax.scatter(
            x,
            y,
            color="firebrick",
            s=s2,
        )
        ax.text(
            x=x - 3,
            y=y + 0.1,
            s="Simple" if i == 0 else "Complex",
            fontsize=fs_text,
            horizontalalignment="right",
            verticalalignment="bottom",
        )
    
    
    ax.set_ylabel(
        "Estimated intrinsic dimensions",
        fontsize=fs_label,
    )
    ax.set_xlabel(
        "Consortia",
        fontsize=fs_label,
    )
    
    ax.set_yticks(
        [2, 4, 6,],
        labels=[2, 4, 6,],
        fontsize=fs_ticks,
    )
    ax.set_yticks(
        [3, 5,],
        minor=True,
    )
    ax.set_xticks(
        [0, 70, 140,],
        labels=[0, 70, 140,],
        fontsize=fs_ticks,
    )
    ax.set_xticks(
        [10, 20, 30, 40, 50, 60, 80, 90, 100, 110, 120, 130,],
        minor=True,
    )
    for spine in ["top", "right",]:
        ax.spines[spine].set_visible(False)
