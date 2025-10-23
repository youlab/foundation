import matplotlib.pyplot as plt
import numpy as np

from applications.config import NOW_TEXT
from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS_CONSORTIA_EXP,
    DIR_CACHE_CONSORTIA_EXP,
)
from figs.utils.colors import get_colors


def plot_panel(
    ax,
    n_panels,
    n_cols,
    y_true,
    y_pred,
    rep,
    y_max,
    p_x,
    p_y,
):
    n_rows = n_panels // n_cols + (0 if ((n_panels % n_cols) == 0) else 1)
    x_step = 1 / n_cols
    y_step = 1 / n_rows
    idx_labels = (n_rows - 1) * n_cols
    colors = get_colors(n=n_panels)

    for i in range(n_panels):
        i_col = i % n_cols
        i_row = n_rows - (i // n_cols) - 1
        a = ax.inset_axes(
            [i_col * x_step + p_x, i_row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
        )
    
        a.plot(
            y_true[i, :],
            lw=2,
            alpha=0.4,
            color=colors[i],
        )
        a.plot(
            y_pred[i, :],
            lw=1,
            alpha=0.9,
            color=colors[i],
        )
        a.plot(
            [127, 127,],
            [0, y_max],
            "k:",
        )
        a.axis("off")
        a.set_ylim(-0.01, y_max+0.01)
        a.set_xlim(-1, 768,)
        if i == idx_labels:
            a.axis("on")
            for spine in ["top", "right",]:
                a.spines[spine].set_visible(False)

            if rep == 546:
                a.set_xticks([0, 384, 768,], labels=[0, 384, 768,], fontsize=14,)
                a.set_yticks([0, 0.5,], labels=[0, 0.5,], fontsize=14,)
                a.set_xlabel("Time", fontsize=16, fontweight="medium",)
                a.set_ylabel("Cell density", fontsize=16, fontweight="medium",)
            else:
                a.set_xticks([0, 384, 768,], labels=[],)
                a.set_yticks([0, 0.5,], labels=[],)
        if (i_row == (n_rows - 1)) and (i_col == 0):
            a.text(
                0,
                y_max + 0.1,
                f"rep {rep}",
                fontsize=10,
                fontweight="normal",
            )


def get_ax():
    fig = plt.figure(
        figsize=(9, 14,),
        layout="tight",
    )
    ax_dict = fig.subplot_mosaic(
        "zzzzzzzzz;yABCDEFGH;yIJKLMNOP;yQRSTUVWX;yYZabcdef;yghijklmn;yopqrstuv;wwwwwwwww",
        sharex=True,
        sharey=True,
        height_ratios=[0.3,18, 11, 8, 11, 8, 2,0.3],
        width_ratios=[0.4, 1, 1, 1, 1, 1, 1, 1, 1,],
    )
    return fig, ax_dict


def main(
    fs_panel=24,
    fw_panel="semibold",
    p_x = 0.01,
    p_y = 0.01,
    n_cols = 2,
):
    fig, ax_dict = get_ax()
    
    i_ax = 0
    all_labels = list(ax_dict.keys())
    ax_labels = []
    for ax_label in all_labels:
        if ax_label in {"w", "x", "y", "z",}:
            ax_dict[ax_label].axis("off")
            continue
        ax_labels.append(ax_label)
    for experiment in ["soil-a", "soil-b", "soil-c", "water-a", "water-b", "water-c",]:
        if experiment == "water-b":
            reps = ["1", "2", "3a", "3b", "4", "5", "6", "7",]
        else:
            reps = ["1", "2", "3", "4", "5", "6", "7", "8",]
        for rep in reps:
            data = np.load(
                DIR_CACHE_CONSORTIA_EXP
                / f"future_outlook_{experiment}-{rep}_latent_latent_{NOW_TEXT}.npz",
            )
            y_true = data["w_test"][0]
            y_pred = data["w_test_pred"][0]
            ax = ax_dict[ax_labels[i_ax]]
            ax.axis("off")
            plot_panel(
                ax=ax,
                n_panels=y_true.shape[0],
                n_cols=n_cols,
                y_true=y_true,
                y_pred=y_pred,
                rep=rep,
                y_max=1,
                p_x=p_x,
                p_y=p_y,
            )
            i_ax += 1
            
        print(data["w_test"].shape)
    for x, y, label in [
        (0.02, 0.97, "A",),
        (0.02, 0.7, "B",),
        (0.02, 0.53, "C",),
        (0.02, 0.4, "D",),
        (0.02, 0.22, "E",),
        (0.02, 0.09, "F",),
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

    fig.text(
        x=0.055,
        y=0.03,
        s="Cell density",
        fontsize=10,
        rotation="vertical",
    )
    fig.text(
        x=0.08,
        y=0.02,
        s="Time",
        fontsize=10,
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_12.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_12.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_12.svg",
    )
    plt.close()
