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

from applications.karlsson.karlsson_dataset import preprocess_data, train_test_split_dark, downsample_train, KarlssonDarkDataLoader
from applications.karlsson.karlsson_plots import plot_reconstruction, plot_per_idx_metrics, plot_per_window_metrics


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



def main(
    data_dir: str = DIR_DATA_KARLSSON,
    data_output_path: str = None,
    data_output_type: str = 'dataframe', # 'dataframe' or 'json'
    input_type: str = 'raw', # 'raw' or 'latent' or 'pca'
    target_type: str = 'raw', # 'raw' or 'latent' or 'pca'
    append_max: bool = True, # append the window maximum to latent and pca features, on both sides
    use_test_indices_at: str = None, # specify name of full_split_indices json storing previously computed test indices, assumed to be in data_dir
    split_level: str = 'per_replicate', # 'per_replicate' or 'per_species'
    train_size: float = 0.8,
    train_downsample_to_n: int = None, # if specified, downsamples the train set to this number of curves
    window_size: int = SEQ_LEN,
    stride: int = 1,
    max_depth: int = 15,
    random_seed: int = 501, # determines full splitting and others
    downsampling_seed: int = None, # determines downsampling only
    output_subdir: str = None, # for sampling size repeated downsampling steps
):
    # set up seed and time
    np.random.seed(random_seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # name the run
    run_name = (
        f'{input_type}_to_{target_type}'
        f'_{split_level}'
        f'_seed{random_seed}'
        f'_s{stride}'
        f'_{timestamp}'
    )

    if output_subdir is None:
        output_dir = Path(DIR_RESULTS) / 'karlsson_forecast' / run_name
    else:
        # the caller owns the whole subtree below karlsson_forecast
        output_dir = (
            Path(DIR_RESULTS)
            / "karlsson_forecast"
            / output_subdir
        )
     
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Karlsson Forecast")
    print(f"    input_type   : {input_type}")
    print(f"    target_type  : {target_type}")
    print(f"    append_max   : {append_max}")
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

    # ----------------------------------------------------------------------------------------------------
    # splitting

    # use previously computed test indices
    if use_test_indices_at:
        full_indices_name = use_test_indices_at
        print(f'Using previously split indices: {full_indices_name}')
        with open(str(data_dir / use_test_indices_at), 'r') as f:
            prev_full_split_indices = json.load(f)
        
        full_train_indices = prev_full_split_indices['train']
        full_test_indices = prev_full_split_indices['test']
    
    # perform a new data split
    else: 
        full_indices_name =  f'full_split_indices_seed{random_seed}.json'
        print(f'Creating new train-test split indices: {full_indices_name}')

        full_train_indices, full_test_indices = train_test_split_dark(
            data=data,
            train_size=train_size,
            split_level=split_level,
        )

        # save indices if split is new
        full_split_indices = {
            'splitting_seed': random_seed,
            'n_train': len(full_train_indices),
            'n_test': len(full_test_indices),
            'train': full_train_indices,
            'test': full_test_indices,
        }

        full_split_indices_path = data_dir / full_indices_name
        with open(full_split_indices_path, 'w') as f:
            json.dump(full_split_indices, f, indent=2)
    
    # downsample train indices if specified
    if train_downsample_to_n:

        # match downsampling seed with overall seed if none specified
        downsampling_seed = downsampling_seed if downsampling_seed is not None else random_seed
        downsampled_train_indices_name = f'downsampled_train_indices_seed{downsampling_seed}.json'
        print(f'Downsampling full train indices to {train_downsample_to_n}, saving to {downsampled_train_indices_name}')

        train_indices = downsample_train(
            n=train_downsample_to_n, 
            full_train_indices=full_train_indices,
            output_path=str(output_dir / downsampled_train_indices_name),
            downsampling_seed=downsampling_seed,
        )

    else:
        downsampled_train_indices_name = ''
        train_indices = full_train_indices
    
    # keet test indices fixed
    test_indices = full_test_indices

    # ----------------------------------------------------------------------------------------------------
    # forecasting

    # prepare data for regression
    loader = KarlssonDarkDataLoader(
        data=data,
        train_indices=train_indices,
        test_indices=test_indices,
        input_type=input_type,
        target_type=target_type,
        window_size=window_size,
        stride=stride,
        append_max=append_max,
        random_seed=random_seed,
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

    # ----------------------------------------------------------------------------------------------------
    # performance evaluation

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
        "output_dir": str(output_dir),
        "random_seed": random_seed,
        "input_type": input_type,
        "target_type": target_type,
        "append_max": append_max,
        "split_level": split_level,
        "window_size": window_size,
        "stride": stride,
        "train_size": train_size,
        "train_downsample_to_n": train_downsample_to_n,
        "max_depth": max_depth,

        "full_indices_name": full_indices_name,
        "downsampled_train_indices_name": downsampled_train_indices_name,
        "downsampling_seed": downsampling_seed,

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

    # ----------------------------------------------------------------------------------------------------
    # plotting

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
    
    except Exception as e:
        print(f'Metrics plotting error: {e}')
    
    results["y_train"] = y_train
    results["y_train_pred"] = y_train_pred

    results["y_test"] = y_test
    results["y_test_pred"] = y_test_pred

    return results