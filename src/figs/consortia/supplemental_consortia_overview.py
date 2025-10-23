import matplotlib.pyplot as plt

from config import DIR_FIGS_MANUSCRIPT
from figs.consortia.sample_focal_community import plot_sample_focal_community
from figs.consortia.simulated_consortia_counts import plot_simulated_consortia_counts
from figs.consortia.estimated_intrinsic_dimensions import plot_consortia_estimated_intrinsic_dimensions


def get_ax(
    figsize=(10, 5,),
    axis_off=True,
):
    fig = plt.figure(
        figsize=figsize,
    )
    ax = fig.subplot_mosaic(
        "A",
    )["A"]
    if axis_off:
        ax.axis("off")
    return fig, ax


def main():
    
    fig, ax = get_ax(
        figsize=(14, 7,),
    )
    fs_label = 24
    fw_label = "semibold"
    
    a = []
    for bounds in [
        [0, 0, 0.3, 1,],
        [0.45, 0.65, 0.55, 0.35,],
        [0.45, 0.0, 0.55, 0.4,],
    ]:
        a.append(
            ax.inset_axes(
                bounds=bounds,
            )
        )

    plot_sample_focal_community(
        ax=a[1],
        fs_label=18,
        fs_ticks=16,
    )
    
    plot_simulated_consortia_counts(
        ax=a[0],
    )
    
    plot_consortia_estimated_intrinsic_dimensions(
        ax=a[2],
    )
    fig.text(
        0.05,
        0.95,
        "A",
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=fs_label,
        fontweight=fw_label,
    )
    fig.text(
        0.4,
        0.95,
        "B",
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=fs_label,
        fontweight=fw_label,
    )
    fig.text(
        0.4,
        0.5,
        "C",
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=fs_label,
        fontweight=fw_label,
    )
    
    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_9.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_9.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_9.svg",
    )
    plt.close()
