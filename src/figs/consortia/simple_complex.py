import numpy as np

from applications.config import NOW_TEXT
from applications.consortia.config import get_consortia_for_plotting
from config import (
    DIR_RESULTS_CONSORTIA,
    MODEL_TYPE,
    Z_DIM,
)
from figs.utils.colors import get_colors


def plot_consortium_b(
    a,
    x,
    ymax,
    label,
    add_labels=False,
):
    n = x.shape[0]
    colors = get_colors(n=n)
    for i in range(n):
        a.plot(
            x[i],
            color=colors[i],
            lw=3,
            alpha=0.8,
        )
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.set_ylim(
        0,
        ymax,
    )
    if add_labels:
        a.set_xticks(
            [0, 350, 700,],
            labels=[0, 350, 700],
            fontsize=14,
            fontweight="medium",
        )
        a.set_yticks(
            [0, 1,],
            labels=[0, 1,],
            fontsize=14,
            fontweight="medium",
        )
        a.set_xlabel(
            "Time",
            fontsize=18,
            fontweight="medium",
        )
        a.set_ylabel(
            "Cell density",
            fontsize=18,
            fontweight="medium",
        )
    else:
        a.set_xticks(
            [0, 350, 700,],
            labels=[],
        )
        a.set_yticks(
            [0, 1,],
            labels=[],
        )
    a.text(
        700,
        0.75,
        label,
        fontsize=18,
        fontweight="medium",
        horizontalalignment="right",
    )


def main(ax):
    ax.axis("off")
    a = ax.inset_axes([0, 0.5, 1, 0.5,])
    b = ax.inset_axes([0, 0, 1, 0.5,])
    ax = [a, b]

    input_type = "latent"
    consortia = get_consortia_for_plotting()
    train_size = 12800
    input_type = "latent"
    tgt_type = "latent"
    cross_val = 0
        
    data0 = np.load(
        DIR_RESULTS_CONSORTIA
        / f"future_outlook_{input_type}_{consortia[0]}_{tgt_type}_{MODEL_TYPE}_{Z_DIM}_{cross_val}_{train_size}_{NOW_TEXT}.npz",
    )

    data1 = np.load(
        DIR_RESULTS_CONSORTIA
        / f"future_outlook_{input_type}_{consortia[1]}_{tgt_type}_{MODEL_TYPE}_{Z_DIM}_{cross_val}_{train_size}_{NOW_TEXT}.npz",
    )

    plot_consortium_b(
        a=ax[0],
        x=data0["w_test"][0],
        ymax=1.05,
        label="simple",
    )

    plot_consortium_b(
        a=ax[1],
        x=data1["w_test"][0],
        ymax=1.05,
        label="complex",
        add_labels=True,
    )
