import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS,
    SEQ_LEN,
)


# cached outputs of applications/karlsson/forecast_sample_size.py (run with plot_only=False)
RESULTS_DIR = DIR_RESULTS / "karlsson_forecast" / "sample_size_analysis_2"

# panel order a-e
REPRESENTATIONS = [
    "raw_to_raw",
    "pca_to_raw",
    "pca_max_to_raw",
    "latent_to_raw",
    "latent_max_to_raw",
]

REPRESENTATION_LABELS = {
    "raw_to_raw": "Raw to Raw",
    "pca_to_raw": "PCA to Raw",
    "pca_max_to_raw": "PCA + Max to Raw",
    "latent_to_raw": "A7X to Raw",
    "latent_max_to_raw": "A7X + Max to Raw",
}

# panel f, same as applications/karlsson/karlsson_plots.py; the variant without the appended max is dashed
REPRESENTATION_COLORS = {
    "raw_to_raw": "tab:gray",
    "pca_to_raw": "tab:orange",
    "pca_max_to_raw": "tab:orange",
    "latent_to_raw": "tab:blue",
    "latent_max_to_raw": "tab:blue",
}

REPRESENTATION_LINESTYLES = {
    "raw_to_raw": "-",
    "pca_to_raw": "--",
    "pca_max_to_raw": "-",
    "latent_to_raw": "--",
    "latent_max_to_raw": "-",
}

# full test set: 768 test trajectories x 2 non-overlapping target windows
N_TEST_WINDOWS = 1536

# index of the first train size plotted in panel f (drops n = 13)
START_IDX = 1

# seed and number of points for the displayed scatter subsample only
SEED = 501
N_POINTS = 100_000

S_DOT = 2.9

LIM = (0, 330)
TICKS = [100, 200, 300]


def main(
    fs_panel,
    fw_panel,
):
    FS_LABEL = 20
    FS_TICKS = 18
    FS_ANNOT = 18
    FS_LEGEND = 14

    predictions = np.load(RESULTS_DIR / "full_train_predictions.npz")
    with open(RESULTS_DIR / "aggregate_results.json", "r") as f:
        aggregate = json.load(f)

    # ax: (2, 3), panels a-e are scatters, panel f is the sample size sweep
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.5))
    ax = ax.ravel()

    for i_panel, representation in enumerate(REPRESENTATIONS):
        y_true = predictions[f"{representation}__y_test"]  # (N_TEST_WINDOWS, SEQ_LEN)
        y_pred = predictions[f"{representation}__y_test_pred"]  # (N_TEST_WINDOWS, SEQ_LEN)
        assert y_true.shape == (N_TEST_WINDOWS, SEQ_LEN), f"{representation}: unexpected y_test shape {y_true.shape}"
        assert y_pred.shape == y_true.shape, f"{representation}: y_test_pred shape {y_pred.shape} != y_test shape {y_true.shape}"

        # global R2, pooled over all test windows and time points
        global_r2 = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
        print(f"{REPRESENTATION_LABELS[representation]}: global R2 = {global_r2:.4f}")

        # random subsample of flattened (window, time) entries, for display only
        rng = np.random.default_rng(SEED)
        idx = rng.choice(y_true.size, size=N_POINTS, replace=False)

        a = ax[i_panel]
        a.scatter(
            y_true.reshape(-1)[idx],
            y_pred.reshape(-1)[idx],
            color="tab:orange",
            alpha=0.05,
            s=S_DOT,
            rasterized=True,
        )
        a.plot(LIM, LIM, "k--", linewidth=1)

        a.text(
            0.05,
            0.95,
            REPRESENTATION_LABELS[representation],
            transform=a.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=FS_ANNOT,
        )
        a.text(
            0.05,
            0.85,
            f"R² = {global_r2:.3f}",
            transform=a.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=FS_ANNOT,
        )

        a.set_xlim(LIM)
        a.set_ylim(LIM)
        a.set_xticks(TICKS)
        a.set_yticks(TICKS)
        a.set_xlabel("True", fontsize=FS_LABEL)
        a.set_ylabel("Predicted", fontsize=FS_LABEL)

    # panel f
    a = ax[5]
    for representation in REPRESENTATIONS:
        rows = [r for r in aggregate if r["representation"] == representation]
        rows = sorted(rows, key=lambda r: r["train_trajectories"])[START_IDX:]

        a.errorbar(
            [r["train_trajectories"] for r in rows],
            [r["global_r2_test_mean"] for r in rows],
            yerr=[r["global_r2_test_std"] for r in rows],
            marker="o",
            markersize=8,
            capsize=4,
            linewidth=1.5,
            color=REPRESENTATION_COLORS[representation],
            linestyle=REPRESENTATION_LINESTYLES[representation],
            label=REPRESENTATION_LABELS[representation],
            alpha=0.7,
        )

    a.set_xticks([0, 1000, 2000])
    a.set_yticks([0.6, 0.8, 1.0])
    a.set_xlabel("Number of train trajectories", fontsize=FS_LABEL)
    a.set_ylabel("Global R²", fontsize=FS_LABEL)
    a.legend(loc="lower right", fontsize=FS_LEGEND)

    for i_panel, a in enumerate(ax):
        a.set_box_aspect(1)
        a.tick_params(labelsize=FS_TICKS)
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)
        a.text(
            -0.3,
            1.1,
            "abcdef"[i_panel],
            transform=a.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=fs_panel,
            fontweight=fw_panel,
        )

    # explicit margins so all labels lie inside the canvas (bbox_inches="tight" clipped the panel f x-label)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.11, top=0.93, wspace=0.45, hspace=0.38)

    os.makedirs(DIR_FIGS_MANUSCRIPT, exist_ok=True)
    plt.savefig(DIR_FIGS_MANUSCRIPT / "fig_6_new.png", dpi=300)
    plt.savefig(DIR_FIGS_MANUSCRIPT / "fig_6_new.pdf", dpi=300)
    plt.savefig(DIR_FIGS_MANUSCRIPT / "fig_6_new.svg", dpi=300)
    plt.close()
    print(f"Saved figure to {DIR_FIGS_MANUSCRIPT / 'fig_6_new.pdf'}")
