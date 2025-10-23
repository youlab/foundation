import numpy as np

from config import DIR_DATA_CHAOTIC
from figs.utils.colors import get_colors


def plot_sample_focal_community(
    ax,
    n_focal = 5,
    fs_ticks=12,
    fs_label=14,
    lw=3,
):
    colors = get_colors(n=n_focal)
    
    t1 = 1500
    t2 = t1 + 600
    y = np.load(
        DIR_DATA_CHAOTIC
        / "chaotic_2024-06-03_y_raw.npy",
    )
    y = y[:, t1:t2]
    y = y[y.sum(axis=1) > 0.1]
    ax.plot(
        y[n_focal:].T,
        color="k",
        alpha=0.2,
    )
    for i in range(n_focal):
        ax.plot(
            y[i],
            color=colors[i],
            alpha=0.9,
            lw=lw,
        )
    
    ax.set_xticks(
        [0, 300, 600,],
        labels=[0, 300, 600,],
        fontsize=fs_ticks,
    )
    ax.set_yticks(
        [0, 1, 2,],
        labels=[0, 1, 2,],
        fontsize=fs_ticks,
    )
    ax.set_xticks(
        [100, 200, 400, 500,],
        minor=True,
    )
    ax.set_yticks(
        [0.25, 0.5, 0.75, 1.25, 1.5, 1.75,],
        minor=True,
    )
    ax.set_ylabel(
        "Cell density",
        fontsize=fs_label,
    )
    ax.set_xlabel(
        "Time",
        fontsize=fs_label,
    )
    for spine in ["top", "right",]:
        ax.spines[spine].set_visible(False)
