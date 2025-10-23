import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from applications.utils.latents import reconstruct
from data.utils import get_data
from ml.utils.load_models import load_default_model


def format_both(a):
    for spine in ["top", "right",]:
        a.spines[spine].set_visible(False)
    a.set_xticks([])
    a.set_yticks([])


def format_curve(a):
    format_both(a=a)
    a.set_xlim(0, 127,)
    a.set_ylim(0, 1.05,)


def format_scatter(a):
    format_both(a=a)
    a.set_xlim(-0.05, 1.05,)
    a.set_ylim(-0.05, 1.05,)
    a.plot(
        [0, 1,],
        [0, 1,],
        "k:",
    )


def main():
    model = load_default_model()
    data_main, idx = get_data(category="all")

    random_noise2 = np.random.rand(128)

    binary1 = np.zeros(128)
    binary1[np.random.rand(128) > 0.5] = 1
    binary2 = np.zeros(128)
    binary2[32:64] = 1
    binary2[96:] = 1
    binary2 *= np.random.rand(128)

    y = [
        data_main["y"][idx["experimental"]["zach_growth_curves_2023-03-27_3_y.npy"]["y_all_i"]],
        random_noise2,
        data_main["y"][idx["experimental"]["kyeri_KeioV3_mcherry_20231022_2hpre_betalactams_384_cefo_y.npy"]["y_all_i"]+90],
        binary1,
        data_main["y"][idx["experimental"]["fujita_microbiome_microbiome_dataset_y.npy"]["y_all_i"]+10],
        binary2,
    ]

    for i in range(len(y)):
        y[i] = y[i] / y[i].max()

    recon = []
    for y_ in y:
        recon.append(
            reconstruct(
                x=y_.reshape(1, 128),
                model=model,
                batch_size=1,
            ).flatten()
        )

    lw=3
    _, ax = plt.subplots(3,6, figsize=(10, 4.5,), constrained_layout=True, sharex=False, sharey=False,)
    ax = ax.ravel()
    for i in range(6):
        ax[i*3].plot(
            y[i],
            color="cornflowerblue",
            lw=lw,
            alpha=0.6,
        )
        ax[i*3+1].plot(
            recon[i],
            color="firebrick",
            lw=lw,
            alpha=0.6,
        )
        ax[i*3+2].scatter(
            y[i],
            recon[i],
            alpha=0.5,
            s=10,
            color="tab:orange",
        )
        r2 = r2_score(
            y[i],
            recon[i],
        )
        ax[i*3+2].text(
            0,
            1,
            r"$R^2=$"+f"{r2:.2f}",
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="left",
        )
        format_curve(ax[i*3])
        format_curve(ax[i*3+1])
        format_scatter(ax[i*3+2])

    for i in [0, 3,]:
        ax[12+i].set_xticks(
            [0, 50, 100],
            labels=[0, 50, 100],
            fontsize=12,
        )
        ax[13+i].set_xticks(
            [0, 50, 100],
            labels=[0, 50, 100],
            fontsize=12,
        )
        
        ax[12+i].set_yticks(
            [0, 1],
            labels=[0,1,],
            fontsize=12,
        )
        
        ax[14+i].set_xticks(
            [0, 1],
            labels=[0, 1,],
            fontsize=12,
        )
        ax[14+i].set_yticks(
            [0, 1],
            labels=[0,1,],
            fontsize=12,
        )
    ax[0].text(
        127,
        0,
        "original",
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        color="cornflowerblue",
        fontweight="medium",
    )
    ax[1].text(
        127,
        0,
        "reconstruction",
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        color="firebrick",
        fontweight="medium",
    )
