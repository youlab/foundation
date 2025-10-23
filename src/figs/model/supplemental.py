import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from applications.utils.latents import reconstruct
from ml.utils.load_models import load_default_model
from config import (
    DIR_FIGS_MANUSCRIPT,
    MODEL_TYPE,
    SEQ_LEN,
)
from figs.model.examples import panel_examples
from data.utils import get_data


def get_original_and_reconstruction(
    n_curves,
):
    model = load_default_model()
    data, _ = get_data()
    data, _ = train_test_split(
        data["y"][data["test_idx"]],
        train_size=n_curves,
        random_state=42,
    )

    return (
        data,
        reconstruct(
            model=model,
            x=data.reshape(
                n_curves,
                SEQ_LEN,
            ),
            batch_size=4_096,
            is_vae=MODEL_TYPE != "MCR",
            is_pca=MODEL_TYPE.find("pca") > -1,
        ),
    )


def main(
    fs_text=11,
    fw_text="bold",
):
    print("Making figure for model supplemental")
    N_CURVES = 100
    FS_LABELS = 18
    FS_TICKS = 16

    (
        original,
        reconstructed,
    ) = get_original_and_reconstruction(
        n_curves=N_CURVES,
    )
    
    plt.ion()
    _, ax = plt.subplots(
        int(np.sqrt(N_CURVES)),
        int(np.sqrt(N_CURVES)),
        figsize=(
            16,
            16,
        ),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    for i, a in enumerate(ax.ravel()):
        panel_examples(
            a=a,
            y_true=original[i],
            y_recon=reconstructed[i],
            legend=False,
        )

    ax[-1, 0].set_xlabel(
        "Time",
        fontsize=FS_LABELS,
    )
    ax[-1, 0].set_ylabel(
        "Cell density",
        fontsize=FS_LABELS,
    )
    ax[-1, 0].set_xticks(
        [0, 100,],
        labels=[0, 100,],
        fontsize=FS_TICKS,
    )
    ax[-1, 0].set_yticks(
        [0, 1.0,],
        labels=[0, 1.0,],
        fontsize=FS_TICKS,
    )

    ax[-1, 0].text(
        128,
        0.4,
        "original",
        color="cornflowerblue",
        fontsize=fs_text,
        fontweight=fw_text,
        horizontalalignment="right",
        alpha=0.8,
    )
    ax[-1, 0].text(
        128,
        0.2,
        "reconstruction",
        color="firebrick",
        fontsize=fs_text,
        fontweight=fw_text,
        horizontalalignment="right",
        alpha=0.8,
    )    

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_2.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_2.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_2.svg",
    )
    plt.close()
