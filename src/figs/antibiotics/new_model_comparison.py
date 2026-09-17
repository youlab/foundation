import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS_ANTIBIOTICS,
    DIR_RESULTS_MODEL_COMPARISON,
)
from figs.utils.colors import get_colors

MODEL_TYPES = [
    "A7X",
    "MCR",
    "MNM",
    "PR",
    "VB",
]
MODEL_LABELS = {
    "PR": "PCA",
}

# Display order of the antibiotic groups in panel C
ANTIBIOTICS = [
    "CIP",
    "GM",
    "SAM",
    "SXT",
]


def main(
    fs_panel,
    fw_panel,
    z_dim=8,
):
    """Draw the reworked supplemental figure S8: two latent dimension sweeps and one bar chart."""
    # panels A and B are unchanged from the original supp_fig_8 and read the archived pivots
    CONFIG = {
        "classify_antibiotic_summary_pivot": {
            "y_label": "Classification accuracy",
            "y_ticks": [0.73, 0.83, 0.93,],
            "raw": 0.877,
        },
        "regress_antibiotic_summary_pivot": {
            "y_label": "Regression accuracy " + r"($R^2$)",
            "y_ticks": [0.67, 0.77, 0.87,],
            "raw": 0.828,
        },
    }

    FS_LABELS = 16
    FS_LEGEND = 14
    FS_TICKS = 14
    X_TICKS = [2, 4, 6, 8, 12, 16, 20, 24, 32,]
    BAR_WIDTH = 0.13

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(
            16,
            5,
        ),
        constrained_layout=True,
    )
    ax = ax.ravel()
    colors = get_colors(n=5)

    for i_ax, file_name in enumerate(CONFIG.keys()):
        a = ax[i_ax].inset_axes([0.15, 0.15, 0.7, 0.7])
        ax[i_ax].axis("off")
        df = pd.read_csv(
            DIR_RESULTS_MODEL_COMPARISON
            / f"{file_name}.csv",
        )

        config = CONFIG[file_name]
        for i, model_type in enumerate(df.model_type.unique()):
            mask = df.model_type == model_type
            # the first row of each model type is the latent one, the raw rows are unused here
            y = df.loc[mask, [str(z) for z in X_TICKS]].to_numpy()[0]
            x = np.array(X_TICKS)[~np.isnan(y)]
            y = np.array(y)[~np.isnan(y)]
            a.plot(
                x,
                y,
                color=colors[i],
                label=MODEL_LABELS.get(model_type, model_type),
                marker="o",
                alpha=0.7,
                lw=3,
            )
        a.plot(
            [min(X_TICKS), max(X_TICKS)],
            [config["raw"], config["raw"]],
            color="r",
            ls="--",
            label="raw",
            lw=3,
        )

        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)

        a.set_ylim(
            config["y_ticks"][0],
            config["y_ticks"][-1],
        )

        a.set_xticks(
            X_TICKS,
            labels=X_TICKS,
            fontsize=FS_TICKS,
        )
        a.set_xlabel(
            "Latent dimensions",
            fontsize=FS_LABELS,
        )
        a.set_yticks(
            config["y_ticks"],
            labels=config["y_ticks"],
            fontsize=FS_TICKS,
        )
        a.set_ylabel(
            config["y_label"],
            fontsize=FS_LABELS,
        )

    a = ax[2].inset_axes([0.15, 0.15, 0.7, 0.7])
    ax[2].axis("off")
    df = pd.read_csv(
        DIR_RESULTS_ANTIBIOTICS
        / f"new_model_comparison_f1_{z_dim}.csv",
    )

    x_centers = np.arange(len(ANTIBIOTICS))
    bar_order = MODEL_TYPES + ["raw",]
    for i, model in enumerate(bar_order):
        means = []
        stds = []
        for antibiotic in ANTIBIOTICS:
            mask = (df.model == model) & (df.antibiotic == antibiotic)
            assert mask.sum() == 1, f"Expected one row for {model} / {antibiotic}, got {mask.sum()}"
            means.append(df.loc[mask, "f1_mean"].to_numpy()[0])
            stds.append(df.loc[mask, "f1_std"].to_numpy()[0])

        a.bar(
            x_centers + (i - 2.5) * BAR_WIDTH,
            means,
            width=BAR_WIDTH,
            yerr=stds,
            color="r" if model == "raw" else colors[i],
            label=MODEL_LABELS.get(model, model),
            alpha=0.7,
            edgecolor="k",
            capsize=3,
        )

    for spine in ["top", "right",]:
        a.spines[spine].set_visible(False)

    a.set_ylim(0, 1.0)
    a.set_xticks(
        x_centers,
        labels=ANTIBIOTICS,
        fontsize=FS_TICKS,
    )
    a.set_xlabel(
        "Resistance type",
        fontsize=FS_LABELS,
    )
    a.set_yticks(
        [0, 0.25, 0.5, 0.75, 1.0,],
        labels=[0, 0.25, 0.5, 0.75, 1.0,],
        fontsize=FS_TICKS,
    )
    a.set_ylabel(
        "F1 score",
        fontsize=FS_LABELS,
    )
    # legend sits entirely outside the panel on the right
    a.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=FS_LEGEND,
    )

    for x, y, label in [
        (0.0, 0.97, "A",),
        (0.33, 0.97, "B",),
        (0.66, 0.97, "C",),
    ]:
        fig.text(
            x=x,
            y=y,
            s=label,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=fs_panel,
            fontweight=fw_panel,
        )

    os.makedirs(DIR_FIGS_MANUSCRIPT, exist_ok=True)
    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / f"supp_fig_8_new.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / f"supp_fig_8_new.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / f"supp_fig_8_new.svg",
    )
    plt.close()
