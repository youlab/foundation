import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS_MODEL_COMPARISON,
)
from figs.utils.colors import get_colors


def main(
    fs_panel,
    fw_panel,
):
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
        "classify_antibiotics_CIP_summary_pivot": {
            "y_label": "CIP F1 score",
            "y_ticks": [0.73, 0.83, 0.93,],
            "raw": 0.879,
        },
        "classify_antibiotics_GM_summary_pivot": {
            "y_label": "GM F1 score",
            "y_ticks": [0.5, 0.6, 0.7,],
            "raw": 0.639,
        },
        "classify_antibiotics_SAM_summary_pivot": {
            "y_label": "SAM F1 score",
            "y_ticks": [0.73, 0.83, 0.93,],
            "raw": 0.879,
        },
        "classify_antibiotics_SXT_summary_pivot": {
            "y_label": "SXT F1 score",
            "y_ticks": [0.7, 0.8, 0.9,],
            "raw": 0.858,
        },
    }

    FS_LABELS = 16
    FS_LEGEND = 14
    FS_TICKS = 14
    X_TICKS = [2, 4, 6, 8, 12, 16, 20, 24, 32,]

    fig, ax = plt.subplots(
        2,
        3,
        figsize=(
            16,
            9,
        ),
        constrained_layout=True,
    )
    ax = ax.ravel()
    colors = get_colors(n=5)

    i_ax = 0
    z_dims = [2, 4, 6, 8, 12, 16, 20, 24, 32,]
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
            y = df.loc[mask, [str(z_dim) for z_dim in z_dims]].to_numpy()[0]
            x = np.array(z_dims)[~np.isnan(y)]
            y = np.array(y)[~np.isnan(y)]
            a.plot(
                x,
                y,
                color=colors[i],
                label=f"{model_type}",
                marker="o",
                alpha=0.7,
                lw=3,
            )
        a.plot(
            [min(z_dims), max(z_dims)],
            [config["raw"], config["raw"]],
            color="r",
            ls="--",
            label="Using raw",
            lw=3,
        )

        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)

        a.set_ylim(
            config["y_ticks"][0],
            config["y_ticks"][-1],
        )
        if i_ax == 2:
            a.legend(
                bbox_to_anchor=(1.05, 1),
                fontsize=FS_LEGEND,
            )

        if i_ax == 3:
            a.set_xticks(
                X_TICKS,
                labels=X_TICKS,
                fontsize=FS_TICKS,
            )
            a.set_xlabel(
                "Latent dimensions",
                fontsize=FS_LABELS,
            )

        else:
            a.set_xticks(
                X_TICKS,
                labels=[],
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
    
    for x, y, label in [
        (0.0, 0.97, "A",),
        (0.3, 0.97, "B",),
        (0.6, 0.97, "C",),
        (0.0, 0.47, "D",),
        (0.3, 0.47, "E",),
        (0.6, 0.47, "F",),
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

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / f"supp_fig_8.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / f"supp_fig_8.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / f"supp_fig_8.svg",
    )
    plt.close()
