import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DIR_RESULTS_ANTIBIOTICS
)


def get_data(
    antibiotic,
    model_name,
):
    return pd.read_csv(
        DIR_RESULTS_ANTIBIOTICS
        / f"classify_antibiotics_{antibiotic}_summary_{model_name}.csv",
        index_col=0,
    )


def main(
    model_name,
    antibiotic_list=[
        "SAM",
        "CIP",
        "GM",
        "SXT",
    ],
    metric_labels=[
        "raw_accuracy",
        "latent_accuracy",
    ],
    ax=None,
    show_legend=True,
    add_formatting=True,
    alpha=0.8,
):
    FS_LABELS = 18
    FS_LEGEND = 12
    FS_TICKS = 14
    FS_TITLE = 20

    LEGEND_LABELS = {
        "raw_accuracy": "Using raw",
        "latent_accuracy": "Using latent",
    }

    ax_existed = not ax is None
    if ax_existed:
        ax = [ax]
    else:
        plt.ion()
        fig, ax = plt.subplots(
            2,
            2,
            figsize=(
                8,
                8,
            ),
            constrained_layout=True,
            sharex=True,
            sharey=False,
        )
        ax = ax.ravel()

    for i_ax, antibiotic in enumerate(antibiotic_list):
        data = get_data(
            antibiotic=antibiotic,
            model_name=model_name,
        )
        for label in metric_labels:
            ax[i_ax].plot(
                data.train_size.to_numpy(),
                data.loc[:, label].to_numpy(),
                label=LEGEND_LABELS[label],
                marker="o",
                alpha=alpha,
            )

        if add_formatting:
            ax[i_ax].set_xticks(
                [0, 1000, 2000, 3000,],
                labels=["0", "1,000", "2,000", "3,000",],
                fontsize=FS_TICKS,
            )
            ax[i_ax].spines["top"].set_visible(False)
            ax[i_ax].spines["right"].set_visible(False)

    if not ax_existed:
        fig.suptitle(f"Classification Accuracy of {data['test_size']:,.0f} Test Curves", fontsize=FS_TITLE,)
        fig.supylabel("F1 Score", fontsize=FS_LABELS,)
        fig.supxlabel("Training Curves", fontsize=FS_LABELS,)
    else:
        if add_formatting:
            ax[0].set_ylabel(f"F1 Score {antibiotic}", fontsize=FS_LABELS,)
            ax[0].set_xlabel("Training Curves", fontsize=FS_LABELS,)

    if show_legend:
        ax[-1].legend(
            fontsize=FS_LEGEND,
            bbox_to_anchor=(1, -0.3),
            loc='lower right',
            ncols=2,
        )

    if not ax_existed:
        plt.savefig(
            DIR_RESULTS_ANTIBIOTICS
            / f"classify_antibiotic_resistance_f1_score_{model_name}.png",
        )

        plt.close()


def plot_resistance_panels(
    axes,
    model_name,
    metric_labels,
    add_legend=[False, False, False, False,],
    fs_labels=16,
    fs_ticks=14,
    fs_text=16,
):
    y_specs = {
        "CIP": {
            "y_lim": (0.68, 0.92,),
            "y_ticks": [0.7, 0.8, 0.9,],
            "y_ticks_minor": [0.75, 0.85,],
        },
        "GM": {
            "y_lim": (0.43, 0.69,),
            "y_ticks": [0.45, 0.55, 0.65,],
            "y_ticks_minor": [0.5, 0.6,],
        },
        "SAM": {
            "y_lim": (0.68, 0.92,),
            "y_ticks": [0.7, 0.8, 0.9,],
            "y_ticks_minor": [0.75, 0.85,],
        },
        "SXT": {
            "y_lim": (0.68, 0.92,),
            "y_ticks": [0.7, 0.8, 0.9,],
            "y_ticks_minor": [0.75, 0.85,],
        },
    }
    for i, antibiotic in enumerate([
        "CIP",
        "SAM",
        "SXT",
        "GM",
    ]):
        a = axes[i]
        main(
            model_name=model_name,
            antibiotic_list=[antibiotic],
            metric_labels=metric_labels,
            ax=a,
            show_legend=add_legend[i],
            add_formatting=False,
            alpha=0.8,
        )
        if i == 2:
            a.set_xticks(
                [0, 4000,],
                labels=[0, "4,000",],
                fontsize=fs_ticks,
            )
            a.set_xlabel(
                "Train size",
                fontsize=fs_labels,
            )
            a.set_ylabel(
                "F1 score",
                fontsize=fs_labels,
            )
        else:
            a.set_xticks(
                [0, 4000,],
                labels=[],
            )
            
        a.set_yticks(
            y_specs[antibiotic]["y_ticks"],
            labels=y_specs[antibiotic]["y_ticks"],
            fontsize=fs_ticks,
        )
        
        a.set_ylim(y_specs[antibiotic]["y_lim"])
        a.set_xticks(
            [1000, 2000, 3000,],
            minor=True,
        )
        a.set_yticks(
            y_specs[antibiotic]["y_ticks_minor"],
            minor=True,
        )
        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)
        a.text(
            x=3_000,
            y=y_specs[antibiotic]["y_ticks_minor"][0],
            s=antibiotic,
            fontsize=fs_text,
            horizontalalignment="center",
            verticalalignment="center",
        )
