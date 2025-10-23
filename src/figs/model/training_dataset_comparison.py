import matplotlib.pyplot as plt
import pandas as pd

from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS_MODEL_COMPARISON,
)
from figs.utils.colors import get_colors


def process_suffix(suffix):
    if suffix in {"all", "experimental", "simulation"}:
        return suffix
    return "all"


def main():
    FS_LABEL = 16
    FS_TICKS = 14
    FS_LEGEND = 12

    colors = get_colors(n=3)

    df = pd.read_csv(
        DIR_RESULTS_MODEL_COMPARISON
        / "reconstruction_all.csv",
        index_col=0,
    )
    df["training_dataset"] = [process_suffix(suffix=val.split("_")[-1]) for val in df.model_name]
    d2 = df.groupby(["training_dataset", "z_dim"]).agg("max", numeric_only=True)
    d3 = d2.reset_index()
    fig, ax = plt.subplots(2, 3, figsize=(10, 5,), tight_layout=True,)
    dataset_names = ["both", "experimental", "simulation",]
    for i, y_val in enumerate(["r2_test_all", "r2_test_exp", "r2_test_sim",]):
        for z_dim in d3.z_dim.unique():
            mask = d3.z_dim == z_dim
            d3.loc[mask, "rank"] = d3.loc[mask, y_val].rank(ascending=False,)
        for j, dataset in enumerate(["all", "experimental", "simulation",]):
            ax[0, i].plot(
                d2.loc[dataset, y_val].index.to_numpy(),
                d2.loc[dataset, y_val].to_numpy(),
                color=colors[j],
                alpha=0.8,
                label=f"Model trained\non {'both' if dataset == 'all' else dataset}",
                marker="o",
            )
            ax[1, i].plot(
                d3.loc[d3.training_dataset == dataset, "z_dim"].to_numpy(),
                d3.loc[d3.training_dataset == dataset, "rank"].to_numpy(),
                color=colors[j],
                alpha=0.8,
                marker="o",
            )
        ax[1, i].invert_yaxis()
        ax[0, i].text(
            8,
            0.9,
            f"Model evaluated\non {dataset_names[i]}",
            fontsize=FS_LEGEND,
        )
    
    ax[0, -1].legend(
        bbox_to_anchor=(1.05, 1.,),
        fontsize=FS_LEGEND,
    )
    for a in ax.ravel():
        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)
        a.set_xticks(
            [2, 4, 6, 8, 12, 16, 24, 32,],
            labels=[],
            fontsize=FS_TICKS,
        )

    for a in ax[0]:
        a.set_yticks(
            [0.9, 0.95, 1.0,],
            labels=[],
            fontsize=FS_TICKS,
        )
        a.set_yticks(
            [0.875, 0.925, 0.975,],
            minor=True,
        )
        a.set_ylim(0.85, 1.01)

    for a in ax[1]:
        a.set_yticks(
            [1, 2, 3,],
            labels=[],
            fontsize=FS_TICKS,
        )

    ax[0, 0].set_yticks(
        [0.9, 0.95, 1.0,],
        labels=[0.9, "", 1.0],
        fontsize=FS_TICKS,
    )
    ax[1, 0].set_yticks(
        [1, 2, 3,],
        labels=[1, 2, 3,],
        fontsize=FS_TICKS,
    )

    ax[0, 0].set_ylabel(
        r"$R^2$",
        fontsize=FS_LABEL,
    )
    ax[1, 0].set_ylabel(
        r"Rank (1 = best)",
        fontsize=FS_LABEL,
    )
    ax[1, 0].set_xlabel(
        "Latent dimensions",
        fontsize=FS_LABEL,
    )
    ax[1, 0].set_xticks(
        [2, 4, 6, 8, 12, 16, 24, 32,],
        labels=[2, "", "", 8, "", 16, 24, 32,],
        fontsize=FS_TICKS,
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_1.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_1.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_1.svg",
    )
    plt.close()
