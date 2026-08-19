"""
Figure R6: test set reconstruction scatters across simulation percentages.

One panel per simulation percentage (0, 10, 20, 40, 80, 100 left to right). Every panel is
scored on the SAME held-out experimental curves, which are identical across all six splits by
construction, so R2 varies only with the training composition and not with the evaluation
target. Simulated held-out curves are excluded for exactly that reason.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from applications.utils.latents import reconstruct
from config import DIR_DATA_PROCESSED, DIR_RESULTS_MODEL_PERC_SIM_SENSITIVITY, SEQ_LEN
from data.utils import get_data
from ml.utils.load_models import load_trained_model

PERC_SIM = [0, 10, 20, 40, 80, 100]

# matched to the karlsson sample size summary figure so panels stay legible at figure width
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 20
ANNOTATION_FONTSIZE = 20

MAX_POINTS = 100_000    # plotted points per panel, identical across panels
SCATTER_SEED = 501


def compute_test_reconstruction(perc, use_cache=True):
    """Reconstruct the held-out experimental curves for one simulation percentage."""
    data_run_dir = f"perc_sim_{perc:03d}"
    model_dir = DIR_RESULTS_MODEL_PERC_SIM_SENSITIVITY / f"a7x_08_perc_sim_{perc:03d}_all_1000epochs"

    if not os.path.isfile(f"{model_dir}/model.pth"):
        print(f"Skipping model directory {model_dir} as it does NOT contain a trained model")
        return None, None, None

    data, _ = get_data(
        run_dir=data_run_dir,
        category="all",
    )

    with open(DIR_DATA_PROCESSED / data_run_dir / "data_config_all.json", "r") as fp:
        data_config = json.load(fp)
    n_experimental = data_config["n_experimental"]

    # experimental rows occupy indices 0 .. n_experimental - 1 of the corpus
    test_idx = data["test_idx"]
    y_true = data["y"][test_idx[test_idx < n_experimental]]    # (61284, SEQ_LEN)

    assert y_true.shape == (61284, SEQ_LEN), f"y_true shape {y_true.shape}"
    assert y_true.dtype == np.float64

    path_reconstruction = model_dir / "reconstruction_test_experimental.npy"
    path_r2 = model_dir / "r2_test_experimental.npy"

    if use_cache and path_reconstruction.exists() and path_r2.exists():
        print(f"perc_sim {perc}: loading cached reconstruction from {model_dir}")
        y_pred = np.load(path_reconstruction)
        r2 = float(np.load(path_r2))
    else:
        print(f"perc_sim {perc}: reconstructing with model from {model_dir}")
        model = load_trained_model(
            model_type="A7X",
            model_dir=model_dir,
        )
        y_pred = reconstruct(
            model=model,
            x=y_true,
            batch_size=4096,
            is_vae=True,
            is_pca=False,
        )
        r2 = r2_score(
            y_true=y_true.flatten(),
            y_pred=y_pred.flatten(),
        )
        np.save(path_reconstruction, y_pred)
        np.save(path_r2, np.array(r2))

    assert y_pred.shape == y_true.shape, f"y_pred shape {y_pred.shape}"
    print(f"perc_sim {perc}: R2 = {r2:.4f}")

    return y_true, y_pred, r2


def draw_reconstruction(ax, y_true, y_pred, perc, r2):
    """Draw one true vs predicted scatter, annotated with its percentage and R2."""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # the same count and seed in every panel, so point density is comparable across panels
    if y_true_flat.size > MAX_POINTS:
        keep_idx = np.random.default_rng(SCATTER_SEED).choice(
            y_true_flat.size,
            size=MAX_POINTS,
            replace=False,
        )
        y_true_flat = y_true_flat[keep_idx]
        y_pred_flat = y_pred_flat[keep_idx]

    ax.scatter(
        y_true_flat,
        y_pred_flat,
        alpha=0.1,
        s=6,
    )
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1,
    )

    # the curves are normalised to their maximum, so the unit box is exact and not a crop
    ax.set_xlim(0.0001, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=TICK_FONTSIZE)

    # no titles, so the percentage and R2 go inside the panel
    ax.text(
        0.05,
        0.95,
        f"{perc}% simulation",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=ANNOTATION_FONTSIZE,
    )
    ax.text(
        0.05,
        0.87,
        f"R² = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=ANNOTATION_FONTSIZE,
    )


def main(use_cache=True):

    fig, axes = plt.subplots(1, len(PERC_SIM), figsize=(30, 5.5))
    y_true_reference = None
    r2s = []

    for panel, perc in enumerate(PERC_SIM):

        y_true, y_pred, r2 = compute_test_reconstruction(
            perc=perc,
            use_cache=use_cache,
        )

        if y_true is None:
            continue

        # all six models must be scored on the identical held-out experimental curves
        if y_true_reference is None:
            y_true_reference = y_true
        else:
            assert np.array_equal(y_true, y_true_reference), (
                f"held-out experimental curves differ at perc_sim {perc}"
            )

        r2s.append(r2)
        ax = axes[panel]
        draw_reconstruction(
            ax=ax,
            y_true=y_true,
            y_pred=y_pred,
            perc=perc,
            r2=r2,
        )

        ax.set_xlabel("True", fontsize=LABEL_FONTSIZE)
        if panel == 0:
            ax.set_ylabel("Predicted", fontsize=LABEL_FONTSIZE)

    fig.tight_layout()

    # save a vector copy alongside the raster one for figure assembly
    output_path = DIR_RESULTS_MODEL_PERC_SIM_SENSITIVITY / "figure_r6_reconstruction_test"
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(
            output_path.with_suffix(ext),
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    print(f"\nSaved Figure R6 to {output_path}.png")
    print("\nSimulation percentage | R2 on held-out experimental curves")
    for perc, r2 in zip(PERC_SIM, r2s):
        print(f"{perc:>21d} | {r2:.4f}")
