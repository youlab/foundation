import pandas as pd

from config import DIR_RESULTS_MODEL_COMPARISON
from figs.utils.colors import get_colors


def plot_model_comparison(
    ax,
    fs_labels=16,
    fs_legend=14,
    fs_ticks=14,
    x_label="Latent dimensions",
    y_label=r"$R^2$",
    legend_loc="lower right",
    y_ticks=[0.7, 0.8, 0.9, 1.0,],
):
    X_TICKS = [2, 4, 6, 8, 12, 16, 20, 24, 32,]

    df = pd.read_csv(
        DIR_RESULTS_MODEL_COMPARISON
        / "reconstruction.csv",
        index_col=0,
    )

    colors = get_colors(n=df.model_type.unique().size)

    for i, model_type in enumerate(["A7X", "MNM", "VB", "MCR", "PR",]):
        mask = df.model_type == model_type
        ax.plot(
            df.loc[mask, "z_dim"].to_numpy(),
            df.loc[mask, "r2_test_all"].to_numpy(),
            color=colors[i],
            ls="-",
            label=f"{model_type}",
            marker="o",
            alpha=0.7,
            lw=4,
            zorder=5-i,
        )

    for spine in ["top", "right",]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc=legend_loc,
        fontsize=fs_legend,
    )
    ax.set_ylim(
        y_ticks[0]-0.04,
        y_ticks[-1]+0.02,
    )

    ax.set_xticks(
        X_TICKS,
        labels=X_TICKS,
        fontsize=fs_ticks,
    )
    ax.set_yticks(
        y_ticks,
        labels=y_ticks,
        fontsize=fs_ticks,
    )

    ax.set_xlabel(
        x_label,
        fontsize=fs_labels,
    )
    ax.set_ylabel(
        y_label,
        fontsize=fs_labels,
    )
