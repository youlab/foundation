from figs.utils.colors import get_colors
from data.utils import get_data


def plot_growth_curves(
    ax,
    fs_label=18,
    fs_ticks=16,
    lw=3,
):
    data, idx = get_data()
    colors = get_colors(n=4)
    ax[0].plot(
        data["y"][idx["experimental"]["zach_growth_curves_2023-03-27_3_y.npy"]["y_all_i"]],
        color=colors[0],
        lw=lw,
    )

    for i in range(3):
        ax[1].plot(
            data["y"][idx["simulation"]["chaotic_2024-06-03-v2_y.npy"]["y_all_i"] + i*100],
            color=colors[i+1],
            lw=lw,
            alpha=0.8,
        )

    ax[0].set_xticks(
        [0, 50, 100,],
        labels=[0, 50, 100,],
        fontsize=fs_ticks,
    )
    ax[0].set_xlabel(
        "Time",
        fontsize=fs_label,
    )
    ax[1].set_xticks(
        [0, 50, 100,],
        labels=[],
    )
    ax[0].set_yticks(
        [0.0, 0.5, 1.0,],
        labels=[0.0, 0.5, 1.0,],
        fontsize=fs_ticks,
    )
    ax[1].set_yticks(
        [0.0, 0.5, 1.0,],
        labels=[],
    )
    ax[0].set_ylabel(
        "Cell density",
        fontsize=fs_label,
    )

    for a in ax:
        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)
