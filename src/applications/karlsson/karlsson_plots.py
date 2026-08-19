from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error

# one cell of 2x3 summary figure
PANEL_FIGSIZE = (6, 6)

# some hardcoded fontsizes for plots
SUMMARY_LABEL_FONTSIZE = 22
SUMMARY_TICK_FONTSIZE = 20
SUMMARY_LEGEND_FONTSIZE = 15
SUMMARY_ANNOTATION_FONTSIZE = 20


def _draw_reconstruction(
    ax,
    y_true,
    y_pred,
    max_points: int = 80000,
    random_state: int = 501,
):
    '''draw the predicted vs actual scatter into a given axes, without labels or title'''

    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)

    # downsample if too many points
    if len(y_true_flat) > max_points:

        rng = np.random.default_rng(random_state)
        keep_idx = rng.choice(
            len(y_true_flat),
            size=max_points,
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

    min_val = min(
        y_true_flat.min(),
        y_pred_flat.min(),
    )

    max_val = max(
        y_true_flat.max(),
        y_pred_flat.max(),
    )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        linewidth=1,
    )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def plot_reconstruction(
    y_true,
    y_pred,
    title: str = "",
    output_path: str = None,
    max_points: int = 80000,
    random_state: int = 501,
):
    '''plot predicted vs actual values of trajectories across all target windows'''

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    _draw_reconstruction(
        ax,
        y_true,
        y_pred,
        max_points=max_points,
        random_state=random_state,
    )

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        for ext in [".png", ".pdf"]:
            fig.savefig(
                output_path.with_suffix(ext),
                dpi=300,
                bbox_inches="tight",
            )

    plt.close(fig)



def plot_per_idx_metrics(
    y_true,
    y_pred,
    title: str = "",
    output_dir: str = None,
):
    """
    Plot per-target-index performance, where index is with respect to each window internally

    For a target window of length w:

        y[:, 0]
        y[:, 1]
        ...
        y[:, w-1]

    compute R2, RMSE, and NRMSE independently at each target index across all target windows

    Returns
    -------
    r2_by_index : ndarray
    rmse_by_index : ndarray
    nrmse_by_index : ndarray
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D targets. Got shape {y_true.shape}")

    window_size = y_true.shape[1]

    r2_per_index = np.zeros(
        window_size,
        dtype=np.float32,
    )
    rmse_per_index = np.zeros(
        window_size,
        dtype=np.float32,
    )
    nrmse_per_index = np.zeros(
        window_size,
        dtype=np.float32,
    )

    # compute metrics
    for idx in range(window_size):

        r2_per_index[idx] = r2_score(
            y_true[:, idx],
            y_pred[:, idx],
        )
        rmse_per_index[idx] = root_mean_squared_error(
            y_true[:, idx],
            y_pred[:, idx],
        )
        range_val = y_true[:, idx].max() - y_true[:, idx].min()
        nrmse_per_index[idx] = rmse_per_index[idx] / range_val if range_val > 0 else 0.0

    # R2 plot
    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        np.arange(window_size),
        r2_per_index,
        linewidth=2,
    )

    ax.set_xlabel("Target index (internal to window)")
    ax.set_ylabel("R2")
    ax.set_title(f"{title}: R2")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for ext in [".png", ".pdf"]:
            fig.savefig(
                output_dir / f"r2_per_index{ext}",
                dpi=300,
                bbox_inches="tight",
            )

    plt.close(fig)

    # RMSE plot
    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        np.arange(window_size),
        rmse_per_index,
        linewidth=2,
    )

    ax.set_xlabel("Target index (internal to window)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{title}: RMSE")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for ext in [".png", ".pdf"]:
            fig.savefig(
                output_dir / f"rmse_per_index{ext}",
                dpi=300,
                bbox_inches="tight",
            )

    plt.close(fig)

    # NRMSE plot
    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        np.arange(window_size),
        nrmse_per_index,
        linewidth=2,
    )

    ax.set_xlabel("Target index (internal to window)")
    ax.set_ylabel("NRMSE")
    ax.set_title(f"{title}: NRMSE")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for ext in [".png", ".pdf"]:
            fig.savefig(
                output_dir / f"nrmse_per_index{ext}",
                dpi=300,
                bbox_inches="tight",
            )

    plt.close(fig)

    return r2_per_index, rmse_per_index, nrmse_per_index



# ====================================================================================================
# PROBLEM - this is plotting all windows as flattened
# ====================================================================================================
def plot_per_window_metrics(
    y_true,
    y_pred,
    title: str = "",
    output_dir: str = None,
):
    """
    Compute and plot per-window metrics
    Each sample corresponds to one input window to target window task

    Returns
    -------
    r2_per_window : ndarray
    rmse_per_window : ndarray
    nrmse_per_window : ndarray
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{y_true.shape} vs {y_pred.shape}"
        )

    n_windows = y_true.shape[0]

    r2_per_window = np.zeros(
        n_windows,
        dtype=np.float32,
    )

    rmse_per_window = np.zeros(
        n_windows,
        dtype=np.float32,
    )

    nrmse_per_window = np.zeros(
        n_windows,
        dtype=np.float32,
    )

    # compute metrics
    for i in range(n_windows):

        r2_per_window[i] = r2_score(
            y_true[i],
            y_pred[i],
        )

        rmse_per_window[i] = root_mean_squared_error(
            y_true[i],
            y_pred[i],
        )

        range_val = y_true[i].max() - y_true[i].min()
        nrmse_per_window[i] = rmse_per_window[i] / range_val if range_val > 0 else 0.0

    window_idx = np.arange(n_windows)

    # R2 plot
    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        window_idx,
        r2_per_window,
        linewidth=1,
    )

    ax.set_xlabel("Forecast window index")
    ax.set_ylabel("R2")
    ax.set_title(f"{title}: R2")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "r2_per_window.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    # RMSE plot
    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        window_idx,
        rmse_per_window,
        linewidth=1,
    )

    ax.set_xlabel("Forecast window index")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{title}: R2")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "rmse_per_window.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    # NRMSE plot
    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        window_idx,
        nrmse_per_window,
        linewidth=1,
    )

    ax.set_xlabel("Forecast window index")
    ax.set_ylabel("NRMSE")
    ax.set_title(f"{title}: NRMSE")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "nrmse_per_window.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    return r2_per_window, rmse_per_window, nrmse_per_window


# ====================================================================================================
# SAMPLE SIZE ANALYSIS
# ====================================================================================================

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
    "latent_to_raw": "Latent to Raw",
    "latent_max_to_raw": "Latent + Max to Raw",
}

# each representation keeps one hue, the variant without the appended max is dashed
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


def _draw_sample_size_metric(
    ax,
    aggregate_results,
    metric,
    ylabel,
    start_idx: int = 0,
    label_fontsize=SUMMARY_LABEL_FONTSIZE,
    tick_fontsize=SUMMARY_TICK_FONTSIZE,
    legend_fontsize=SUMMARY_LEGEND_FONTSIZE,
):
    '''draw one metric against number of training trajectories into a given axes'''

    for representation in REPRESENTATIONS:
        rows = [r for r in aggregate_results if r["representation"] == representation]
        rows = sorted(rows, key=lambda x: x["train_trajectories"])

        if len(rows) == 0:
            continue

        x = [r["train_trajectories"] for r in rows[start_idx:]]
        y = [r[f"{metric}_mean"] for r in rows[start_idx:]]
        yerr = [r[f"{metric}_std"] for r in rows[start_idx:]]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            markersize=8,
            capsize=4,
            linewidth=1.5,
            color=REPRESENTATION_COLORS[representation],
            linestyle=REPRESENTATION_LINESTYLES[representation],
            label=REPRESENTATION_LABELS[representation],
            alpha=0.7,
        )

    ax.set_xlabel("Number of train trajectories", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=legend_fontsize)

    ax.set_box_aspect(1)


def _plot_sample_size_metric(
    aggregate_results,
    metric,
    ylabel,
    output_path,
    start_idx: int = 0,
):
    '''plots one metric against number of training trajectories, formatted as a summary figure panel'''

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    _draw_sample_size_metric(
        ax,
        aggregate_results=aggregate_results,
        metric=metric,
        ylabel=ylabel,
        start_idx=start_idx,
    )

    fig.tight_layout()
    output_path = Path(output_path)
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(
            output_path.with_suffix(ext),
            dpi=300,
        )

    plt.close(fig)


def plot_sample_size_results(
    aggregate_results,
    output_dir,
    start_idx: int = 0,
):
    '''plots sample size cross validation analysis in form of metrics against downsampled number of train trajectories'''

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        (
            "r2_test",
            "R²",
            "r2.png",
        ),
        (
            "rmse_test",
            "RMSE",
            "rmse.png",
        ),
        (
            "nrmse_test",
            "NRMSE",
            "nrmse.png",
        ),
        (
            "global_r2_test",
            "Global R²",
            "global_r2.png",
        ),
        (
            "global_rmse_test",
            "Global RMSE",
            "global_rmse.png",
        ),
        (
            "global_nrmse_test",
            "Global NRMSE",
            "global_nrmse.png",
        ),
    ]

    for metric, ylabel, filename in metrics:
        _plot_sample_size_metric(
            aggregate_results=aggregate_results,
            metric=metric,
            ylabel=ylabel,
            output_path=output_dir / filename,
            start_idx=start_idx,
        )


def plot_sample_size_summary_figure(
    predictions,
    aggregate_results,
    output_path,
    start_idx: int = 0,
):
    '''2x3 summary figure, five full train set reconstruction scatters plus the global R2 sweep'''
    # predictions maps representation name to (y_test, y_test_pred) at the full train set

    fig, axes = plt.subplots(2, 3, figsize=(3 * PANEL_FIGSIZE[0], 2 * PANEL_FIGSIZE[1]))

    for panel, representation in enumerate(REPRESENTATIONS):

        ax = axes[panel // 3, panel % 3]
        if representation not in predictions:
            ax.axis("off")
            continue

        y_true, y_pred = predictions[representation]

        _draw_reconstruction(
            ax,
            y_true,
            y_pred,
        )

        # global R2 on the full pre-downsampling test sets
        global_r2 = r2_score(
            np.asarray(y_true).reshape(-1),
            np.asarray(y_pred).reshape(-1),
        )

        ax.text(
            0.05,
            0.95,
            REPRESENTATION_LABELS[representation],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=SUMMARY_ANNOTATION_FONTSIZE,
        )

        ax.text(
            0.05,
            0.87,
            f"R² = {global_r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=SUMMARY_ANNOTATION_FONTSIZE,
        )

        ax.set_xlabel("True", fontsize=SUMMARY_LABEL_FONTSIZE)
        ax.set_ylabel("Predicted", fontsize=SUMMARY_LABEL_FONTSIZE)
        ax.tick_params(labelsize=SUMMARY_TICK_FONTSIZE)

    _draw_sample_size_metric(
        axes[1, 2],
        aggregate_results=aggregate_results,
        metric="global_r2_test",
        ylabel="Global R²",
        start_idx=start_idx,
        label_fontsize=SUMMARY_LABEL_FONTSIZE,
        tick_fontsize=SUMMARY_TICK_FONTSIZE,
        legend_fontsize=SUMMARY_LEGEND_FONTSIZE,
    )

    fig.tight_layout()

    output_path = Path(output_path)
    for ext in [".png", ".pdf"]:
        fig.savefig(
            output_path.with_suffix(ext),
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)