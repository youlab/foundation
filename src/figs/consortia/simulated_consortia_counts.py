import numpy as np
import pandas as pd

from config import DIR_RESULTS_CONSORTIA


def plot_simulated_consortia_counts(
    ax,
    bar_height = 0.7,
    fs_labels = 14,
    fs_legend = 14,
    fs_ticks=10,
    fs_data_labels = 10,
):
    df = pd.read_csv(
        DIR_RESULTS_CONSORTIA
        / "dims.csv",
        index_col=0,
    )
    df = df.loc[[file_path.find("oscillation") == -1 for file_path in df.file_path], :]
    df = df.loc[[file_path.find("binary") == -1 for file_path in df.file_path], :]
    df["background"] = [int(val.split("/")[-1].split("_B")[1].split("_")[0]) for val in df.file_path]
    df["focal"] = [int(val.split("/")[-1].split("_T")[1].split("_")[0]) for val in df.file_path]
    df["total"] = df["background"] + df.focal
    df_plot = df.groupby(["total", "focal"]).count().reset_index()
    y_pos = np.arange(df_plot.shape[0]) + 1
    
    total_bars = ax.barh(
        y_pos,
        df_plot.loc[:, "total"],
        height=bar_height,
        color="k",
        label='Total',
        alpha=0.4,
        edgecolor="k",
    )
    focal_bars = ax.barh(
        y_pos,
        df_plot.focal,
        height=bar_height,
        color="cornflowerblue",
        label='Focal',
        alpha=1,
        edgecolor="k",
    )

    labels = []
    for i, bar in enumerate(total_bars):
        width = bar.get_width()  # Length of the bar (i.e., the value)
        # The y-position is the center of the bar
        y_center = bar.get_y() + bar.get_height() / 2
        # Place the label a little to the right of the bar's end
        count = df_plot.file_path.iloc[i]
        labels.append(
            f"{df_plot.focal.iloc[i]}/{df_plot.total.iloc[i]}"
        )
        ax.text(
            width + 3,
            y_center,
            f'{count} consortia' if count > 1 else f"{count} consortium", 
            va='center',
            ha='left',
            fontsize=fs_data_labels,
            color='black',
            fontweight="medium",
        )

    ax.set_yticks(
        y_pos,
        labels=labels,
        fontsize=fs_ticks,
    )
    ax.set_xticks(
        [0, 50, 100,],
        labels=[0, 50, 100],
        fontsize=fs_ticks,
    )
    ax.set_xticks(
        [25, 75, 125],
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
    ax.set_ylabel(
        "Focal / Total",
        fontsize=fs_labels,
    )
