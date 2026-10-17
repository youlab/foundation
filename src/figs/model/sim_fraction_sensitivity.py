import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from applications.utils.latents import reconstruct
from config import (
    DIR_DATA_PROCESSED,
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS_MODEL_PERC_SIM_SENSITIVITY,
    SEQ_LEN,
)
from data.utils import get_data
from ml.utils.load_models import load_trained_model


# percentage of the simulation corpus added to the fixed experimental training set, one column each
PERC_SIM = [0, 10, 20, 40, 80, 100]

# experimental train / test curves, identical across all six splits by construction
N_EXP = {
    "train": 245133,
    "test": 61284,
}

# seed and number of points for the displayed scatter subsample only; R2 uses all points
SEED = 0
N_POINTS = 100_000


def main():
    '''
    Rebuttal Figure R6 / Supplementary Figure S2: true vs. A7X-reconstructed values for the experimental train (top) and
    test (bottom) curves, for models trained with increasing fractions of the simulation corpus. Reconstructions are cached
    next to each model; missing ones (train) are computed with the trained model on first run.
    '''
    FS_TICKS = 18
    FS_LABEL = 20
    FS_TITLE = 22
    FS_COLUMN_TITLE = 16
    FS_R2 = 20

    rng = np.random.default_rng(SEED)
    y_reference = {}

    # ax: (2, 6), row 0 = train, row 1 = test
    fig, ax = plt.subplots(2, len(PERC_SIM), figsize=(18, 6.5))

    for k, perc in enumerate(PERC_SIM):
        data_run_dir = f"perc_sim_{perc:03d}"
        model_dir = DIR_RESULTS_MODEL_PERC_SIM_SENSITIVITY / f"a7x_08_perc_sim_{perc:03d}_all_1000epochs"

        data, _ = get_data(run_dir=data_run_dir, category="all")
        with open(DIR_DATA_PROCESSED / data_run_dir / "data_config_all.json", "r") as fp:
            n_experimental = json.load(fp)["n_experimental"]

        for i_row, (split, color) in enumerate([("train", "tab:blue"), ("test", "tab:orange")]):
            # experimental rows occupy indices 0 ... n_experimental - 1 of the corpus
            idx_split = data[f"{split}_idx"]
            y = data["y"][idx_split[idx_split < n_experimental]]  # (N_EXP[split], SEQ_LEN)
            assert y.shape == (N_EXP[split], SEQ_LEN), f"perc_sim {perc} {split}: unexpected data shape {y.shape}"
            assert y.dtype == np.float64, f"perc_sim {perc} {split}: unexpected dtype {y.dtype}"

            # every column must be scored on the identical experimental curves
            if split not in y_reference:
                y_reference[split] = y
            else:
                assert np.array_equal(y, y_reference[split]), f"perc_sim {perc}: experimental {split} curves differ from perc_sim {PERC_SIM[0]}"

            # load cached reconstruction or compute and cache it
            path_reconstruction = model_dir / f"reconstruction_{split}_experimental.npy"
            path_r2 = model_dir / f"r2_{split}_experimental.npy"
            if path_reconstruction.exists() and path_r2.exists():
                print(f"perc_sim {perc} {split}: loading cached reconstruction from {model_dir}")
                rec = np.load(path_reconstruction)
                r2_cached = float(np.load(path_r2))
            else:
                print(f"perc_sim {perc} {split}: reconstructing with model from {model_dir}")
                model = load_trained_model(model_type="A7X", model_dir=model_dir)
                rec = reconstruct(
                    model=model,
                    x=y,
                    batch_size=4096,
                    is_vae=True,
                    is_pca=False,
                )
                r2_cached = r2_score(y_true=y.flatten(), y_pred=rec.flatten())
                np.save(path_reconstruction, rec)
                np.save(path_r2, np.array(r2_cached))
                del model

            assert rec.shape == y.shape, f"perc_sim {perc} {split}: reconstruction shape {rec.shape} != data shape {y.shape}"

            # global R2, pooled over all curves and all time points
            r2 = r2_score(y_true=y.flatten(), y_pred=rec.flatten())
            assert np.isclose(r2, r2_cached), f"perc_sim {perc} {split}: recomputed R2 {r2} != cached R2 {r2_cached}"

            # random subsample of flattened (curve, time) entries, covering all time indices
            idx = rng.choice(y.size, size=N_POINTS, replace=False)

            a = ax[i_row, k]
            a.scatter(
                y.reshape(-1)[idx],
                rec.reshape(-1)[idx],
                color=color,
                alpha=0.05,
                s=1,
                rasterized=True,
            )
            a.plot([0, 1], [0, 1], "k:", lw=2)
            a.text(
                0.05,
                0.92,
                r"$R^2$" + f" = {r2:.3f}",
                transform=a.transAxes,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=FS_R2,
            )
            print(f"perc_sim {perc} {split}: R2 = {r2:.4f} (n_curves = {y.shape[0]})")

            del y, rec

        del data

        ax[0, k].set_title(f"{perc}% of simulations", fontsize=FS_COLUMN_TITLE, pad=20)

    for a in ax.ravel():
        a.set_box_aspect(1)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_xticks([0, 0.5, 1], labels=[])
        a.set_yticks([0, 0.5, 1], labels=[])
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)

    # tick labels and axis labels only on the bottom-left panel
    ax[1, 0].set_xticks([0, 0.5, 1], labels=[0, 0.5, 1], fontsize=FS_TICKS)
    ax[1, 0].set_yticks([0, 0.5, 1], labels=[0, 0.5, 1], fontsize=FS_TICKS)
    ax[1, 0].set_xlabel("True", fontsize=FS_LABEL)
    ax[1, 0].set_ylabel("Predicted", fontsize=FS_LABEL)

    for i_row, label in enumerate(["Train", "Test"]):
        ax[i_row, -1].text(
            1.12,
            0.5,
            label,
            transform=ax[i_row, -1].transAxes,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=FS_TITLE,
            fontweight="bold",
        )

    plt.subplots_adjust(wspace=0.25, hspace=0.1)

    os.makedirs(DIR_FIGS_MANUSCRIPT, exist_ok=True)
    plt.savefig(DIR_FIGS_MANUSCRIPT / "supp_fig_2_new.png", dpi=300, bbox_inches="tight")
    plt.savefig(DIR_FIGS_MANUSCRIPT / "supp_fig_2_new.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(DIR_FIGS_MANUSCRIPT / "supp_fig_2_new.svg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {DIR_FIGS_MANUSCRIPT / 'supp_fig_2_new.pdf'}")
