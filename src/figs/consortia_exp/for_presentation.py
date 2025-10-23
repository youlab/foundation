import matplotlib.pyplot as plt
import numpy as np

from applications.consortia_exp.config import get_config
from applications.consortia_exp.data import get_df_x

from config import DIR_FIGS_MANUSCRIPT
from figs.presentation import get_ax
    

def plot_strains_in_all_eight(
    ax,
    fs_labels=14,
    fs_ticks=12,
):
    def format_axis(
        ax,
    ):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_ylim(0, 1,)
        ax.set_xlim(0, 110,)
        ax.set_xticks([])
        ax.set_yticks([])

    df, x = get_df_x(x_min=0)
    experiment = "water-a"
    strain_def, _ = get_config(source_file=experiment)
    mask_experiment = df.file == experiment

    df_temp = df.loc[mask_experiment, :].copy()
    x_temp = x[mask_experiment, :]

    for i_ax, rep in enumerate(df_temp.rep.unique()):
        mask = df_temp.rep == rep
        mask_strain = np.array([strain in strain_def for strain in df_temp[mask].strain])
        for i in range(mask_strain.size):
            ax[i_ax].plot(
                x_temp[mask][i],
                color="cornflowerblue" if mask_strain[i] else "k",
                alpha=0.7 if mask_strain[i] else 0.4,
            )
        format_axis(
            ax=ax[i_ax],
        )
    ax[-2].set_xticks(
        [0, 55, 110,],
        labels=[0, 55, 110,],
        fontsize=fs_ticks,
    )
    ax[-2].set_yticks(
        [0, 0.5, 1,],
        labels=[0.0, 0.5, 1.0,],
        fontsize=fs_ticks,
    )

    ax[-2].set_ylabel(
        "Fraction",
        fontsize=fs_labels,
    )
    ax[-2].set_xlabel(
        "Time",
        fontsize=fs_labels,
    )


def plot_focal_counts(
    ax,
    bar_height = 0.8,
    fs_labels = 14,
    fs_legend = 14,
    fs_data_labels = 12,
):
    df, _ = get_df_x(x_min=0)
    focals = []
    totals = []
    experiments = []
    for experiment in df.file.unique():
        strain_def, _ = get_config(source_file=experiment)
        mask_experiment = df.file == experiment

        df_temp = df.loc[mask_experiment, :].copy()
        mask_strain = np.array([strain in strain_def for strain in df_temp[mask_experiment].strain.unique()])
        experiments.append(experiment)
        focals.append(mask_strain.sum())
        totals.append(mask_strain.size)
        print(experiment, mask_strain.sum(), mask_strain.size, f"{mask_strain.sum() / mask_strain.size * 100:.2f}",)

    y_pos = np.arange(len(totals)) + 1

    ax.barh(
        y_pos,
        totals,
        height=bar_height,
        color="k",
        label='Total',
        alpha=0.6,
        edgecolor="k",
    )
    focal_bars = ax.barh(
        y_pos,
        focals,
        height=bar_height,
        color="cornflowerblue",
        label='Focal',
        alpha=1,
        edgecolor="k",
    )


    for i, bar in enumerate(focal_bars):
        width = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2
        ax.text(
            width + 3,
            y_center,
            f'{focals[i]} focal', 
            va='center',
            ha='left',
            fontsize=fs_data_labels,
            color='white',
            fontweight="bold",
        )

    ax.invert_yaxis()
    ax.set_yticks(
        y_pos,
        labels=experiments,
        fontsize=fs_labels,
    )
    ax.set_xticks(
        [0, 50, 100, 150,],
        labels=[0, 50, 100, 150,],
        fontsize=fs_labels,
    )
    ax.set_xticks(
        [25, 75, 125,],
        minor=True,
    )
    for spine in ["top", "right",]:
        ax.spines[spine].set_visible(False)
    ax.legend(
        loc="lower right",
        fontsize=fs_legend,
    )
    ax.set_xlabel(
        "Count",
        fontsize=fs_labels,
    )


def main():
    ax = get_ax()
    ax = ax.inset_axes(
        [0.1, 0.1, 0.8, 0.8,],
    )
    plot_focal_counts(
        ax,
        bar_height = 0.8,
        fs_labels = 14,
        fs_legend = 14,
        fs_data_labels = 12,
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_11.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_11.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_11.png",
    )
    plt.close()
