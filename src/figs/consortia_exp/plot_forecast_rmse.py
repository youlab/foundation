import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DIR_RESULTS_CONSORTIA_EXP,
    DIR_SRC,
)
from figs.utils.colors import get_colors


def load_data():
    df = pd.read_csv(
        DIR_RESULTS_CONSORTIA_EXP
        / "consortia_exp_forecast_rmse_epsilon.csv",
        index_col=0,
    )
    df["source"] = [val[0] for val in df.experiment.str.split("-")]
    df["media"] = [val[1] for val in df.experiment.str.split("-")]
    return df


def main(ax=None):
    df = load_data()
    colors = get_colors(n=4)
    
    ax_existed = ax is not None
    if not ax_existed:
        plt.ion()
        _, ax = plt.subplots(1, 1, figsize=(9, 5,), constrained_layout=True, sharex=True,)
        add_data_labels = True
    else:
        add_data_labels = False

    FS_TICKS = 14
    FS_LEGEND = 12
    FS_LABELS = 16

    bar_height = 0.4
    categories = []
    raws = []
    latents = []
    y_tick_labels = []
    for experiment in df.experiment.unique():
        categories.append(experiment)
        raws.append(df.loc[(df.experiment == experiment) & (df.input_type == "raw"), "rmse"].iloc[0])
        latents.append(df.loc[(df.experiment == experiment) & (df.input_type == "latent"), "rmse"].iloc[0])
        y_tick_labels.append(
            experiment
        )

    y_pos = np.arange(len(categories)) + 1
    latent_bars = ax.barh(
        y_pos - bar_height/2,
        latents,
        height=bar_height,
        color=colors[0],
        label='Latent',
        alpha=0.6,
        edgecolor="k",
    )
    raw_bars = ax.barh(
        y_pos + bar_height/2,
        raws,
        height=bar_height,
        color=colors[2],
        label='Raw',
        alpha=0.6,
        edgecolor="k",
    )
    if add_data_labels:
        for bars in [raw_bars, latent_bars,]:
            for bar in bars:
                width = bar.get_width()
                y_center = bar.get_y() + bar.get_height() / 2
                ax.text(
                    width + 0.01,
                    y_center,
                    f'{width:.3f}', 
                    va='center',
                    ha='left',
                    fontsize=12,
                    color='black',
                )
    ax.invert_yaxis()
    ax.set_yticks(
        y_pos,
        labels=y_tick_labels,
        fontsize=FS_TICKS,
    )
    ax.legend(
        loc="lower left",
        fontsize=FS_LEGEND,
    )
    ax.set_xticks(
        [0, 0.2, 0.4],
        labels=[0, 0.2, 0.4,],
        fontsize=FS_TICKS,
    )
    for spine in ["top", "right",]:
        ax.spines[spine].set_visible(False)

    ax.set_xlabel(
        "RMSE",
        fontsize=FS_LABELS,
    )
    ax.set_ylabel(
        "Experiment",
        fontsize=FS_LABELS,
    )
    if not ax_existed:
        plt.savefig(
            DIR_SRC
            / "results"
            / "figs"
            / "consortia_exp"
            / "forecast_rmse.png",
        )
        plt.close()
