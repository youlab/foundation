from pathlib import Path
from datetime import datetime
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

from config import (
    DIR_DATA_KARLSSON,
    DIR_RESULTS,
    SEQ_LEN,
)

from applications.karlsson.karlsson_dataset import preprocess_data, train_test_split_dark, KarlssonDarkDataLoader

def compute_metrics(y_true, y_pred):
    '''
    Returns
    -------
    r2 : float
        Mean per window R2.

    rmse : float
        Mean per window RMSE.

    global_r2 : float
        R2 on flattened arrays.

    global_rmse : float
        RMSE on flattened arrays.

    nrmse : float
        Mean per window NRMSE (normalized by max-min).

    global_nrmse : float
        NRMSE on flattened arrays (normalized by max-min).
    '''

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # for a single output, just in case
    if y_true.ndim == 1:
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        range_val = float(y_true.max() - y_true.min())
        nrmse = rmse / range_val if range_val > 0 else 0.0
        return r2, rmse, r2, rmse, nrmse, nrmse

    # per window metrics
    r2_per_dim = []
    rmse_per_dim = []
    nrmse_per_dim = []

    for dim in range(y_true.shape[1]):

        r2_per_dim.append(
            r2_score(
                y_true[:, dim],
                y_pred[:, dim],
            )
        )

        rmse_per_dim.append(
            root_mean_squared_error(
                y_true[:, dim],
                y_pred[:, dim],
            )
        )

        range_val = y_true[:, dim].max() - y_true[:, dim].min()
        nrmse_val = rmse_per_dim[-1] / range_val if range_val > 0 else 0.0
        nrmse_per_dim.append(nrmse_val)

    # averaged over all windows
    r2 = float(np.mean(r2_per_dim))
    rmse = float(np.mean(rmse_per_dim))
    nrmse = float(np.mean(nrmse_per_dim))

    # global metrics
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    global_r2 = float(r2_score(
        y_true_flat,
        y_pred_flat,
    ))

    global_rmse = float(root_mean_squared_error(
        y_true_flat,
        y_pred_flat,
    ))

    global_range = y_true_flat.max() - y_true_flat.min()
    global_nrmse = global_rmse / global_range if global_range > 0 else 0.0

    return r2, rmse, nrmse, global_r2, global_rmse, global_nrmse


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


def main(
    data_dir: str = DIR_DATA_KARLSSON,
    data_output_path: str = None,
    data_output_type: str = 'dataframe', # 'dataframe' or 'json'
    input_type: str = 'raw', # 'raw' or 'latent' or 'pca'
    target_type: str = 'raw', # 'raw' or 'latent' or 'pca'
    split_level: str = 'per_replicate', # 'per_replicate' or 'per_species'
    train_size: float = 0.8,
    window_size: int = SEQ_LEN,
    stride: int = 1,
    max_depth: int = 15,
    random_seed: int = 501,
):
    # set up seed and time
    np.random.seed(random_seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # name the run
    run_name = (
        f'{input_type}_to_{target_type}'
        f'_{split_level}'
        f'_seed{random_seed}'
        f'_s{stride}'
        f'_{timestamp}'
    )

    output_dir = Path(DIR_RESULTS) / 'karlsson_forecast' / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Karlsson Forecast")
    print(f"    input_type   : {input_type}")
    print(f"    target_type  : {target_type}")
    print(f"    split_level  : {split_level}")
    print(f"    window_size  : {window_size}")
    print(f"    stride       : {stride}")
    print(f"    random_seed  : {random_seed}")
    print("=" * 80)

    # load data
    data = preprocess_data(
        data_dir=data_dir,
        output_path=data_output_path,
        output_type=data_output_type,
    )

    # split data
    train_indices, test_indices = train_test_split_dark(
        data=data,
        train_size=train_size,
        split_level=split_level,
    )

    # prepare data for regression
    loader = KarlssonDarkDataLoader(
        data=data,
        train_indices=train_indices,
        test_indices=test_indices,
        input_type=input_type,
        target_type=target_type,
        window_size=window_size,
        stride=stride,
    )

    X_train, y_train, X_test, y_test = loader.get_regression_data()

    print(
        f"    X_train={X_train.shape}, "
        f"    y_train={y_train.shape}"
    )

    print(
        f"    X_test={X_test.shape}, "
        f"    y_test={y_test.shape}"
    )

    # train regressor
    regressor = ExtraTreesRegressor(
        random_state=random_seed,
        criterion="friedman_mse",
        max_depth=max_depth,
        n_estimators=100,
    )
    print("Training ExtraTreesRegressor...")
    regressor.fit(X_train, y_train)

    # test
    print("Generating predictions...")
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    # report metrics
    r2_train, rmse_train, nrmse_train, global_r2_train, global_rmse_train, global_nrmse_train = compute_metrics(y_train, y_train_pred)
    r2_test, rmse_test, nrmse_test, global_r2_test, global_rmse_test, global_nrmse_test = compute_metrics(y_test, y_test_pred)

    # save results
    results = {
        "timestamp": timestamp,
        "random_seed": random_seed,
        "input_type": input_type,
        "target_type": target_type,
        "split_level": split_level,
        "window_size": window_size,
        "stride": stride,
        "train_size": train_size,
        "max_depth": max_depth,

        "n_train_trajectories": len(train_indices),
        "n_test_trajectories": len(test_indices),

        "n_train_windows": int(X_train.shape[0]),
        "n_test_windows": int(X_test.shape[0]),

        "r2_train": float(r2_train),
        "rmse_train": float(rmse_train),
        "nrmse_train": float(nrmse_train),

        "r2_test": float(r2_test),
        "rmse_test": float(rmse_test),
        "nrmse_test": float(nrmse_test),

        "global_r2_train": float(global_r2_train),
        "global_rmse_train": float(global_rmse_train),
        "global_nrmse_train": float(global_nrmse_train),

        "global_r2_test": float(global_r2_test),
        "global_rmse_test": float(global_rmse_test),
        "global_nrmse_test": float(global_nrmse_test),
    }

    results_path = output_dir / 'results.json'

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("Results")
    print(
        f"    Train: R²={r2_train:.4f}, "
        f"    RMSE={rmse_train:.6f}, "
        f"    NRMSE={nrmse_train:.6f}"
    )
    print(
        f"    Test : R²={r2_test:.4f}, "
        f"    RMSE={rmse_test:.6f}, "
        f"    NRMSE={nrmse_test:.6f}"
    )
    print("=" * 80)
    print(f"Saved to: {output_dir}")

    # plot results
    train_plots_dir = output_dir / 'train'
    test_plots_dir = output_dir / 'test'
    train_plots_dir.mkdir(parents=True, exist_ok=True)
    test_plots_dir.mkdir(parents=True, exist_ok=True)

    plot_reconstruction(
        y_train,
        y_train_pred,
        title=f'Train Reconstruction: Global R2={global_r2_train:.3f}',
        output_path=str(train_plots_dir / 'reconstruction.png'),
    )

    plot_reconstruction(
        y_test,
        y_test_pred,
        title=f'Test Reconstruction: Global R2={global_r2_test:.3f}',
        output_path=str(test_plots_dir / 'reconstruction.png'),
    )

    try:
        r2_per_index_train, rmse_per_index_train, nrmse_per_index_train = plot_per_idx_metrics(
            y_train,
            y_train_pred,
            title='Train Metrics per Window Internal Index',
            output_dir=str(train_plots_dir),
        )

        r2_per_index_test, rmse_per_index_test, nrmse_per_index_test = plot_per_idx_metrics(
            y_test,
            y_test_pred,
            title='Test Metrics per Window Internal Index',
            output_dir=str(test_plots_dir),
        )

        r2_per_window_train, rmse_window_index_train, nrmse_per_window_train = plot_per_window_metrics(
            y_train,
            y_train_pred,
            title='Train Metrics per Windows',
            output_dir=str(train_plots_dir),
        )

        r2_per_window_test, rmse_per_window_test, nrmse_per_window_test = plot_per_window_metrics(
            y_test,
            y_test_pred,
            title='Test Metrics per Window',
            output_dir=str(test_plots_dir),
        )
    except Exception as e:
        print(f'Plotting error: {e}')

    return results