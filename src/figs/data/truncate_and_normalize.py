import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from applications.utils.data import get_t_cols
from config import DIR_DATA_ZACH
from figs.utils.colors import get_colors


def format_left(
    a,
    add_labels=False,
):
    a.set_xlim(0, 184)
    a.set_ylim(0, 1.1,)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    if add_labels:
        a.set_xticks(
            [0, 60, 120, 180,],
            labels=[0, 60, 120, 180,],
            fontsize=14,
            fontweight="medium",
        )
        a.set_yticks(
            [0, 0.5, 1.0,],
            labels=[0, 0.5, 1.0,],
            fontsize=14,
            fontweight="medium",
        )
        a.set_xlabel(
            "Time points",
            fontsize=16,
            fontweight="medium",
        )
        a.set_ylabel(
            "Cell density",
            fontsize=16,
            fontweight="medium",
        )
    else:
        a.set_xticks(
            [0, 60, 120, 180,],
            labels=[],
        )
        a.set_yticks(
            [0, 0.5, 1.0,],
            labels=[],
        )


def format_right(
    a,
    add_labels=False,
):
    a.set_xlim(0, 127)
    a.set_ylim(0, 1.1,)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    if add_labels:
        a.set_xticks(
            [0, 60, 120,],
            labels=[0, 60, 120,],
            fontsize=14,
            fontweight="medium",
        )
    else:
        a.set_xticks(
            [0, 60, 120,],
            labels=[],
        )
    a.set_yticks(
        [0, 0.5, 1.0,],
        labels=[],
    )


def main():
    df = pd.read_csv(
        DIR_DATA_ZACH
        / "growth_curves_2023-03-27_3.csv",
        index_col=0,
    )
    t_cols, _ = get_t_cols(p=df)
    x = df.loc[:, t_cols].to_numpy()
    x_plot = x[[val in {0, 26, 65, 74} for val in np.arange(x.shape[0])], :]

    _, ax = plt.subplots(
        4,
        2,
        figsize=(
            6,
            5,
        ),
        constrained_layout=True,
        width_ratios=[185, 128],
    )
    colors = get_colors(n=4)
    x_norm = x_plot[:, :128]
    x_norm = x_norm / x_norm.max(axis=1).reshape(-1, 1)

    for i in range(4):
        ax[i, 0].plot(
            x_plot[i],
            color=colors[i],
            lw=3,
        )
        ax[i, 1].plot(
            x_norm[i],
            color=colors[i],
            lw=3,
        )
        format_left(
            a=ax[i, 0],
            add_labels=i == 3,
        )
        format_right(
            a=ax[i, 1],
            add_labels=i == 3,
        )
