import matplotlib.pyplot as plt

from config import DIR_FIGS_MANUSCRIPT
from ml.utils.load_models import load_default_model
from figs.model.examples_beta import plot_examples_beta
from figs.model.model_comparison import plot_model_comparison
from figs.model.model_reconstruction_accuracy import plot_model_reconstruction_accuracy
from figs.model.reconstruction_pipeline import plot_reconstruction_pipeline
from figs.model.twod_by_source import plot_twod_by_source
from data.utils import get_data


def get_axes():
    fig = plt.figure(
        figsize=(16, 9,),
        layout="tight",
    )
    ax_dict = fig.subplot_mosaic(
        "ACD;BCD;EEE",
        height_ratios=[1, 1, 2],
        width_ratios=[1, 1, 1,],
    )
    return fig, ax_dict


def main(
    fs_panel,
    fw_panel,
):
    model = load_default_model()
    data, idx = get_data(category="all")
    fig, ax_dict = get_axes()

    ax = ax_dict["A"].inset_axes([0.1, 0.0, 0.9, 0.9,])
    ax_dict["A"].axis("off")
    plot_reconstruction_pipeline(
        ax=ax,
        data=data,
        idx=idx,
        model=model,
        fs_labels=16,
        p_x=5,
        p_y=0.02,
        lw=5,
        ax_bounds_1=[0.05, 0.05, 0.3, 0.8,],
        ax_bounds_2=[0.73, 0.05, 0.3, 0.8,],
        fs_text=13,
        fw_text="bold",
        fw_labels="medium",
        fs_vae_label=13,
        fw_vae_label="semibold",
    )

    ax = ax_dict["B"].inset_axes([0.1, 0.0, 0.9, 1.0,])
    ax_dict["B"].axis("off")
    plot_model_reconstruction_accuracy(
        ax=ax,
        fs_ticks=14,
        fs_label=16,
        p_x=0.04,
        p_y=0.00,
        x_r2=0.1,
        y_r2=0.9,
        fs_r2=16,
        horizontal_alignment="left",
        x_dataset=0.6,
        use_cache=False,
    )

    ax = ax_dict["C"]
    plot_twod_by_source(
        ax=ax,
        model=model,
        fs_text=16,
        fw_text="semibold",
        x_text=180,
        y_text=-140,
        vertical_alignment="center",
        horizontal_alignment="center",
        p_x=0.05,
        p_y=0.01,
        left=-0.05,
    )

    ax = ax_dict["D"].inset_axes([0.1, 0.1, 0.8, 0.8,])
    ax_dict["D"].axis("off")
    plot_model_comparison(
        ax=ax,
    )

    ax = ax_dict["E"].inset_axes([0.02, 0.1, 0.98, 0.9,])
    ax_dict["E"].axis("off")
    plot_examples_beta(
        ax=ax,
        data=data,
        idx=idx,
        model=model,
        fs_text=14,
        fw_text="bold",
    )

    for x, y, label in [
        (0.01, 0.97, "A",),
        (0.01, 0.73, "B",),
        (0.01, 0.45, "C",),
        (0.33, 0.97, "D",),
        (0.69, 0.97, "E",),
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
        / "fig_2.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_2.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_2.svg",
    )
    plt.close()
