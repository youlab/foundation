import matplotlib.pyplot as plt

from config import DIR_FIGS_MANUSCRIPT
from figs.data.data_sources import plot_data_sources
from figs.data.three_types import plot_three_types


def get_axes():
    fig = plt.figure(
        figsize=(16, 9,),
        layout="tight",
    )

    return fig, fig.subplot_mosaic(
        "AD;BD;CD",
        width_ratios=[1, 4,],
    )


def main(
    fw_panel,
    fs_panel,
):
    fig, ax_dict = get_axes()
    ax = [
        ax_dict[label].inset_axes(
            [0.17, 0.1, 0.8, 0.8,],
        ) for label in [
            "A",
            "B",
            "C",
        ]
    ]
    for label in ["A", "B", "C",]:
        ax_dict[label].axis("off")
    plot_three_types(
        ax=ax,
        colormap_name="Blues",
    )

    ax = ax_dict["D"].inset_axes(
        [0.05, 0, 0.6, 1,],
    )
    ax_dict["D"].axis("off")
    plot_data_sources(
        ax=ax,
        p_y=0.01,
        n_cols=4,
    )

    for x, y, label in [
        (0.01, 0.97, "A",),
        (0.01, 0.65, "B",),
        (0.01, 0.33, "C",),
        (0.25, 0.97, "D",),
        (0.25, 0.16, "E",),
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
        / "fig_1.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_1.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_1.svg",
    )
    plt.close()
