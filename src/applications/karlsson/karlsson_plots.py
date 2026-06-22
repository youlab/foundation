from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error


def plot_reconstruction(
    y_true,
    y_pred,
    title: str = "",
    output_path: str = None,
    max_points: int = 80000,
    random_state: int = 501,
):
    '''plot predicted vs actual values of trajectories across all target windows'''

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

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

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

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    if output_path:
        fig.savefig(
            output_path,
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
        fig.savefig(
            output_dir / "r2_per_index.png",
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
        fig.savefig(
            output_dir / "rmse_per_index.png",
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
        fig.savefig(
            output_dir / "nrmse_per_index.png",
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