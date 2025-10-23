import matplotlib.pyplot as plt
import numpy as np

from applications.config import NOW_TEXT
from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS_CONSORTIA,
    DIR_CACHE_CONSORTIA,
    MODEL_TYPE,
    Z_DIM,
)
from figs.utils.colors import get_colors


def get_ax():
    fig = plt.figure(
        figsize=(9, 14,),
        layout="tight",
    )
    ax_dict = fig.subplot_mosaic(
        "ABCDQU;EFGHRV;IJKLSW;MNOPTX",
        sharex=True,
        sharey=True,
        height_ratios=[5, 5, 8, 8,],
    )

    return fig, ax_dict


def load_data():
    dir_simple = DIR_CACHE_CONSORTIA
    dir_complex = DIR_CACHE_CONSORTIA
    cross_val = 0
    # TODO: make sure the caption accurately represents this.
    train_size_sm = 12800
    train_size_lg = 102400
    data_simple_sm = np.load(dir_simple / f"future_outlook_latent_simple_latent_{MODEL_TYPE}_{Z_DIM}_{cross_val}_{train_size_sm}_{NOW_TEXT}.npz")
    data_simple_lg = np.load(dir_simple / f"future_outlook_latent_simple_latent_{MODEL_TYPE}_{Z_DIM}_{cross_val}_{train_size_lg}_{NOW_TEXT}.npz")
    data_complex_sm = np.load(dir_complex / f"future_outlook_latent_complex_latent_{MODEL_TYPE}_{Z_DIM}_{cross_val}_{train_size_sm}_{NOW_TEXT}.npz")
    data_complex_lg = np.load(dir_complex / f"future_outlook_latent_complex_latent_{MODEL_TYPE}_{Z_DIM}_{cross_val}_{train_size_lg}_{NOW_TEXT}.npz")

    datasets = {
        "simple_sm": data_simple_sm,
        "simple_lg": data_simple_lg,
        "complex_sm": data_complex_sm,
        "complex_lg": data_complex_lg,
    }
    return datasets


def main(
    fs_panel,
    fw_panel,
    p_x=0.01,
    p_y=0.01,
    n_cols=1,

):
    fig, ax_dict = get_ax()
    datasets = load_data()
    
    keys = {
        "A": ("simple_sm", -1, 5,),
        "B": ("simple_sm", -2, 5,),
        "C": ("simple_sm", -3, 5,),
        "D": ("simple_sm", -4, 5,),
        "Q": ("simple_sm", -5, 5,),
        "U": ("simple_sm", -6, 5,),
        "E": ("simple_lg", -1, 5,),
        "F": ("simple_lg", -2, 5,),
        "G": ("simple_lg", -3, 5,),
        "H": ("simple_lg", -4, 5,),
        "R": ("simple_lg", -5, 5,),
        "V": ("simple_lg", -6, 5,),
        "I": ("complex_sm", -1, 8,),
        "J": ("complex_sm", -2, 8,),
        "K": ("complex_sm", -3, 8,),
        "L": ("complex_sm", -4, 8,),
        "S": ("complex_sm", -5, 8,),
        "W": ("complex_sm", -6, 8,),
        "M": ("complex_lg", -1, 8,),
        "N": ("complex_lg", -2, 8,),
        "O": ("complex_lg", -3, 8,),
        "P": ("complex_lg", -4, 8,),
        "T": ("complex_lg", -5, 8,),
        "X": ("complex_lg", -6, 8,),
    }

    for key in keys.keys():
        ax = ax_dict[key]
        ax.axis("off")
        data = datasets[keys[key][0]]
        idx = keys[key][1]
        n_panels = keys[key][2]

        y_true = data["w_test"]
        y_pred = data["w_test_pred"]
        y_max = max(y_true.max(), y_pred.max(),)
        
        colors = get_colors(n=n_panels)
        n_rows = n_panels // n_cols
        x_step = 1 / n_cols
        y_step = 1 / n_rows
        idx_labels = (n_rows - 1) * n_cols

        for i in range(n_panels):
            i_col = i % n_cols
            i_row = n_rows - (i // n_cols) - 1
            a = ax.inset_axes(
                [i_col * x_step + p_x, i_row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
            )
        
            a.plot(
                y_true[idx, i, :],
                lw=4,
                alpha=0.4,
                color=colors[i],
            )
            a.plot(
                y_pred[idx, i, :],
                lw=2,
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

                if idx == -1:
                    a.set_xticks([0, 384, 768,], labels=[0, 384, 768,], fontsize=14,)
                    y_tick_label = int(y_max * 10) / 10
                    a.set_yticks([0, y_tick_label,], labels=[0, y_tick_label,], fontsize=14,)
                    a.set_xlabel("Time", fontsize=16, fontweight="medium",)
                    a.set_ylabel("Cell density", fontsize=16, fontweight="medium",)
                else:
                    a.set_xticks([0, 384, 768,], labels=[],)
                    a.set_yticks([0, 0.5,], labels=[],)
            if (i_row == (n_rows - 1)) and (i_col == 0):
                a.text(
                    0,
                    y_max + 0.1,
                    f"sample {-idx}",
                    fontsize=12,
                    fontweight="normal",
                )

    for x, y, label in [
        (0.02, 0.99, "A",),
        (0.02, 0.78, "B",),
        (0.02, 0.58, "C",),
        (0.02, 0.28, "D",),
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
        / "supp_fig_10.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_10.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_10.svg",
    )
    plt.close()
