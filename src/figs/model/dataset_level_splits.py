import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from config import (
    DIR_DATA_PROCESSED,
    DIR_FIGS_MANUSCRIPT,
    SEQ_LEN,
)
from data.utils import get_data


# dataset-level splits, in the order shown as Split 1-6 (chronological by run timestamp)
SPLIT_TIMESTAMPS = [
    "2026-05-05T01:11:54.987230",
    "2026-05-05T01:16:08.370504",
    "2026-05-05T01:20:52.269355",
    "2026-05-05T01:25:29.368284",
    "2026-05-05T23:07:34.714356",
    "2026-05-10T19:44:45.339589",
]

# seed and number of points for the displayed scatter subsample only; R2 uses all points
SEED = 0
N_POINTS = 100_000


def main():
    '''
    Rebuttal Figure R1 / Supplementary Figure S3: true vs. A7X-reconstructed values for train (top) and test (bottom)
    curves of the six dataset-level splits. Uses cached reconstructions from plot_model_reconstruction_accuracy.
    '''
    FS_TICKS = 18
    FS_LABEL = 20
    FS_TITLE = 22
    FS_R2 = 20

    rng = np.random.default_rng(SEED)

    # ax: (2, 6), row 0 = train, row 1 = test
    fig, ax = plt.subplots(2, len(SPLIT_TIMESTAMPS), figsize=(18, 6.5))

    for k, ts in enumerate(SPLIT_TIMESTAMPS):
        data_run_dir = Path("dataset_level_splitting_analysis") / f"run_{ts}"
        res_dir = DIR_DATA_PROCESSED / data_run_dir / f"a7x_08_run_{ts}_all_1000epochs"

        data, _ = get_data(run_dir=data_run_dir, category="all")
        y_train = data["y"][data["train_idx"]]  # (n_train, SEQ_LEN)
        y_test = data["y"][data["test_idx"]]  # (n_test, SEQ_LEN)
        del data

        rec_train = np.load(res_dir / "reconstruction_train.npy", mmap_mode="r")  # (n_train, SEQ_LEN)
        rec_test = np.load(res_dir / "reconstruction_test.npy", mmap_mode="r")  # (n_test, SEQ_LEN)
        r2_cached = np.load(res_dir / "r2s.npy")  # [r2_train, r2_test]

        for i_row, (y, rec, color) in enumerate(
            [
                (y_train, rec_train, "tab:blue"),
                (y_test, rec_test, "tab:orange"),
            ]
        ):
            assert y.ndim == 2 and y.shape[1] == SEQ_LEN, f"Split {k + 1}: unexpected data shape {y.shape}"
            assert rec.shape == y.shape, f"Split {k + 1}: reconstruction shape {rec.shape} != data shape {y.shape}"

            # global R2, pooled over all curves and all time points (same definition as plot_model_reconstruction_accuracy)
            r2 = r2_score(y_true=y.flatten(), y_pred=np.asarray(rec).flatten())
            assert np.isclose(r2, r2_cached[i_row]), f"Split {k + 1}: recomputed R2 {r2} != cached R2 {r2_cached[i_row]}"

            # random subsample of flattened (curve, time) entries, covering all time indices
            idx = rng.choice(y.size, size=N_POINTS, replace=False)

            a = ax[i_row, k]
            a.scatter(
                y.reshape(-1)[idx],
                np.asarray(rec).reshape(-1)[idx],
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
            print(f"Split {k + 1} {'train' if i_row == 0 else 'test'}: R2 = {r2:.4f} (n_curves = {y.shape[0]})")

        del y_train, y_test, rec_train, rec_test

        ax[0, k].set_title(f"Split {k + 1}", fontsize=FS_TITLE, pad=20)

    for a in ax.ravel():
        a.set_box_aspect(1)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_xticks([0, 1], labels=[])
        a.set_yticks([0, 1], labels=[])
        a.set_xticks([0.25, 0.5, 0.75], minor=True)
        a.set_yticks([0.25, 0.5, 0.75], minor=True)
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)

    # tick labels and axis labels only on the bottom-left panel
    ax[1, 0].set_xticks([0, 1], labels=[0, 1], fontsize=FS_TICKS)
    ax[1, 0].set_yticks([0, 1], labels=[0, 1], fontsize=FS_TICKS)
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
    plt.savefig(DIR_FIGS_MANUSCRIPT / "supp_fig_3_new.png", dpi=300, bbox_inches="tight")
    plt.savefig(DIR_FIGS_MANUSCRIPT / "supp_fig_3_new.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(DIR_FIGS_MANUSCRIPT / "supp_fig_3_new.svg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {DIR_FIGS_MANUSCRIPT / 'supp_fig_3_new.pdf'}")
