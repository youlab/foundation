import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from applications.antibiotics.helena.inputs import (
    get_inputs,
    get_train_and_test_sets,
)
from config import DIR_CACHE_HELENA


def plot_cla_experiment():
    x, y_all, y_class, y_conc, encoder = get_inputs(
        normalize_concentration=False,
        log_concentration=False,
    )

    labels = encoder.categories_[0]

    label = "CLA"
    mask_class = y_class == np.where(labels == label)[0][0]

    plt.ion()
    fig, ax = plt.subplots(
        2,
        len(np.unique(y_conc[mask_class])) // 2,
        figsize=(10, 3,),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ax = ax.ravel()
    colors = ["b", "r", "g", "k", "m", "y", "c",]

    for j, val in enumerate(sorted(np.unique(y_conc[mask_class]).tolist())):
        mask_conc = mask_class & (y_conc == val)
        ax[j].plot(
            y_all[mask_conc, :].T,
            alpha=0.1,
            color=colors[j % len(colors)],
        )
        ax[j].set_title(f"{val:.1f} ug/mL")
        ax[j].spines["top"].set_visible(False)
        ax[j].spines["right"].set_visible(False)
    fig.supylabel("Normalized Reading (a.u.)")
    fig.supxlabel("Time (a.u.)")
    fig.suptitle(f"Different {label} concentrations")
    plt.savefig(
        DIR_CACHE_HELENA
        / f"helena-cla.png",
    )
    plt.show()
    plt.close()


def plot_sul_experiment():
    x, y_all, y_class, y_conc, encoder = get_inputs(
        normalize_concentration=False,
        log_concentration=False,
    )

    labels = encoder.categories_[0]

    label = "SUL"
    mask_class = y_class == np.where(labels == label)[0][0]

    plt.ion()
    fig, ax = plt.subplots(
        2,
        len(np.unique(y_conc[mask_class])) // 2,
        figsize=(10, 3,),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ax = ax.ravel()
    colors = ["b", "r", "g", "k", "m", "y", "c",]

    for j, val in enumerate(sorted(np.unique(y_conc[mask_class]).tolist())):
        mask_conc = mask_class & (y_conc == val)
        ax[j].plot(
            y_all[mask_conc, :].T,
            alpha=0.1,
            color=colors[j % len(colors)],
        )
        ax[j].set_title(f"{val:.1f} ug/mL")
        ax[j].spines["top"].set_visible(False)
        ax[j].spines["right"].set_visible(False)
    fig.supylabel("Normalized Reading (a.u.)")
    fig.supxlabel("Time (a.u.)")
    fig.suptitle(f"Different {label} concentrations")
    plt.savefig(
        DIR_CACHE_HELENA
        / f"helena-sul.png",
    )
    plt.show()
    plt.close()


def plot_taz_experiment():
    x, y_all, y_class, y_conc, encoder = get_inputs(
        normalize_concentration=False,
        log_concentration=False,
    )

    labels = encoder.categories_[0]

    label = "TAZ"
    mask_class = y_class == np.where(labels == label)[0][0]

    plt.ion()
    fig, ax = plt.subplots(
        2,
        len(np.unique(y_conc[mask_class])) // 2,
        figsize=(10, 3,),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ax = ax.ravel()
    colors = ["b", "r", "g", "k", "m", "y", "c",]

    for j, val in enumerate(sorted(np.unique(y_conc[mask_class]).tolist())):
        mask_conc = mask_class & (y_conc == val)
        ax[j].plot(
            y_all[mask_conc, :].T,
            alpha=0.1,
            color=colors[j % len(colors)],
        )
        ax[j].set_title(f"{val:.1f} ug/mL")
        ax[j].spines["top"].set_visible(False)
        ax[j].spines["right"].set_visible(False)
    fig.supylabel("Normalized Reading (a.u.)")
    fig.supxlabel("Time (a.u.)")
    fig.suptitle(f"Different {label} concentrations")
    plt.savefig(
        DIR_CACHE_HELENA
        / f"helena-taz.png",
    )
    plt.show()
    plt.close()


def plot_latent_representations():
    x, y_all, y_class, y_conc, encoder = get_inputs(
        normalize_concentration=False,
        log_concentration=False,
    )
    labels = encoder.categories_[0]
    COLORS = ["b", "r", "g", "k", "m", "y", "c",]
    
    plt.ion()
    fig, ax = plt.subplots(
        3,
        3,
        figsize=(12, 12,),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    
    axes = {}
    for i, label in enumerate(np.unique(labels)):
        print(label, np.where(labels == label)[0][0])
        mask_class = y_class == np.where(labels == label)[0][0]
        axes[i] = {}
        ax[i, 0].set_title(f"Inhibitor = {label}")
        ax[i, 1].set_title(f"Latent Representation")
        ax[i, 2].set_title(f"Latent Pairwise for OD600")
        for j, val in enumerate(sorted(np.unique(y_conc[mask_class]).tolist())):
            mask_conc = mask_class & (y_conc == val)
            legend_label = [f"{val:.1f}"]
            legend_label.extend(["__nolegend__"] * (mask_conc.sum() - 1))
    
            ax[i, 0].plot(
                y_all[mask_conc, :].T,
                alpha=0.3,
                color=COLORS[j%len(COLORS)],
                label=legend_label,
            )
            for w in range(3):
                ax[i, 1].plot(
                    np.arange(8) + w * 8,
                    x[mask_conc, w*9:w*9+8].T,
                    alpha=0.3,
                    color=COLORS[j%len(COLORS)],
                    label=legend_label,
                )
            for q, vals in enumerate([
                (5, 1,),
                (6, 4,),
                (7, 3,),
                (0, 2,),
            ]):
                k, m = vals
                if q not in axes[i]:
                    left = 0.7 + (q % 2) * 0.15
                    bottom = 0.02 + i * 0.33
                    bottom += 0. if q > 1 else 0.15
                    w = 0.12
                    h = 0.12
                    axes[i][q] = fig.add_axes([left, bottom, w, h,])
                    axes[i][q].set_xticks([])
                    axes[i][q].set_yticks([])
                    axes[i][q].set_title(f"x{k} vs x{m}")
                axes[i][q].scatter(
                    x[mask_conc, k],
                    x[mask_conc, m],
                    alpha=0.3,
                    color=COLORS[j % len(COLORS)],
                )
    
            ax[i, 2].set_xticks([])
            ax[i, 2].set_yticks([])
            for spine in ["top", "right", "bottom", "left",]:
                ax[i, 2].spines[spine].set_visible(False)

    ax = ax.ravel()
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    plt.savefig(
        DIR_CACHE_HELENA
        / f"helena-latent-rep.png",
    )
    plt.show()
    plt.close()
