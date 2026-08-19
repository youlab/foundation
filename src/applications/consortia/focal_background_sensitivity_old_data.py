from pathlib import Path
from datetime import datetime
import json
import logging
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from scipy.interpolate import interp1d
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

from applications.consortia.consortia_sim_dataset import (
    train_test_split_simulations,
    ConsortiaSimDataLoader,
)
from applications.consortia.focal_background_sensitivity import aggregate_results
from applications.karlsson.forecast import compute_metrics
from applications.karlsson.karlsson_plots import plot_reconstruction, plot_per_idx_metrics
from ml.utils.load_models import load_default_model

from config import (
    SEQ_LEN,
    Z_DIM,
)

logger = logging.getLogger(__name__)

# the datasets the manuscript itself used, e.g. bgLV_B15_T5_fixed.txt, B is the number of background
# populations and T the number of focal populations stored in the file
DATASET_PATTERN = re.compile(
    r"(bgLV|dgLV)_B(\d+)_T(\d+)_fixed\.txt"
)

# how many of the stored focal populations the regressor observes, keyed by community size, only
# bgLV at N=20 (5 focal stored) and dgLV at N=100 (8 focal stored) exist in this data
OBSERVED_COUNTS = {
    20: (1, 2, 3, 4, 5),
    100: (1, 2, 4, 6, 8),
}

METRIC_NAMES = [
    "r2_test",
    "rmse_test",
    "nrmse_test",
    "global_r2_test",
    "global_rmse_test",
    "global_nrmse_test",
]


def parse_dataset_name(dataset_path):
    '''pull the simulation type, background size and focal size out of a legacy dataset filename'''

    match = DATASET_PATTERN.fullmatch(Path(dataset_path).name)

    if match is None:
        raise ValueError(
            f'Unrecognised dataset filename {Path(dataset_path).name}, '
            'expected e.g. bgLV_B15_T5_fixed.txt'
        )

    simulation = match.group(1)
    background_size = int(match.group(2))
    focal_size = int(match.group(3))

    # the community is the focal populations plus the background ones
    community_size = background_size + focal_size
    return simulation, community_size, focal_size


def load_txt_dataset(file_path, focal_size: int, n_sim: int = 10000):
    '''read one legacy text dataset and fold it into shape (n_sim, focal_size, n_time)'''

    file_path = Path(file_path)

    table = pd.read_csv(
        file_path,
        sep=r'\s+',
        header=None,
        dtype=np.float32,
    )

    raw = table.to_numpy()

    # rows are simulation major, row = simulation * focal_size + focal
    if raw.shape[0] != (n_sim * focal_size):
        raise ValueError(
            f'{file_path.name} has {raw.shape[0]} rows, expected {n_sim} simulations '
            f'times {focal_size} focal populations.'
        )

    n_time = raw.shape[1]

    dataset = raw.reshape(n_sim, focal_size, n_time)
    assert dataset.shape == (n_sim, focal_size, n_time)
    assert dataset.dtype == np.float32

    logger.info(
        f'Loaded {file_path.name} as {dataset.shape} '
        f'({dataset.nbytes / 1024 ** 2:.1f} MB)'
    )

    return dataset


def interpolate_dataset(dataset, interp_len: int):
    '''interpolate a (n_sim, n_focal, n_time) array onto interp_len time points'''

    n_sim, n_focal, n_time = dataset.shape

    # the interpolation grid is built from integer sample positions, so upsampling only
    if interp_len < n_time:
        raise ValueError(
            f'interp_len ({interp_len}) must be at least the number of simulated time points ({n_time}).'
        )

    # this is the convention used by the legacy RawDataset and by load_simulation_dataset, the integer cast on x1 is part of it
    x2 = np.arange(interp_len).astype(float)
    x1 = np.linspace(0, interp_len - 1, n_time).astype(int).astype(float)

    # interp1d works along the last axis, so flatten the simulation and focal axes together
    f = interp1d(x1, dataset.reshape(-1, n_time))
    interpolated = f(x2)

    interpolated[interpolated > 1] = 1
    interpolated[interpolated < 0] = 0

    interpolated = interpolated.reshape(n_sim, n_focal, interp_len).astype(np.float32)
    assert interpolated.shape == (n_sim, n_focal, interp_len)

    return interpolated


def run_single_dataset(
    raw_dataset, # (n_sim, focal_size, n_time) float32, parsed once by main()
    simulation: str,
    community_size: int,
    focal_size: int,
    result_directory,
    observed_indices, # focal populations the regressor observes
    target_indices, # focal populations the regressor has to predict
    data_file: str = None, # name of the text file raw_dataset came from
    simulation_indices=None, # subsample drawn once by main(), never drawn here
    subsample_reference: str = None, # name of the file holding simulation_indices
    input_type: str = 'raw', # 'raw' or 'latent' or 'pca'
    target_type: str = 'raw', # 'raw' or 'latent' or 'pca'
    interp_len: int = SEQ_LEN * 6,
    window_size: int = SEQ_LEN,
    stride: int = 32,
    append_max: bool = True, # append the segment maximum to latent and pca features
    train_size: float = 0.8,
    split_seed: int = 501,
    max_depth: int = 12,
    n_estimators: int = 25,
    n_jobs: int = None,
    encoder=None, # pass a preloaded model to avoid reloading it per run
    make_plots: bool = True,
):
    '''run the forecasting analysis on one legacy dataset for one observed subset and split'''

    result_directory = Path(result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    observed_indices = np.asarray(observed_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)

    logger.info("=" * 80)
    logger.info("Consortia focal/background forecast, original manuscript data")
    logger.info(f"    dataset      : {data_file}")
    logger.info(f"    n_observed   : {len(observed_indices)} of {focal_size} stored focal")
    logger.info(f"    observed     : {observed_indices.tolist()}")
    logger.info(f"    target       : {target_indices.tolist()}")
    logger.info(f"    input_type   : {input_type}")
    logger.info(f"    target_type  : {target_type}")
    logger.info(f"    append_max   : {append_max}")
    logger.info(f"    interp_len   : {interp_len}")
    logger.info(f"    window_size  : {window_size}")
    logger.info(f"    stride       : {stride}")
    logger.info(f"    split_seed   : {split_seed}")
    logger.info("=" * 80)

    # ----------------------------------------------------------------------------------------------------
    # subsetting and interpolation

    n_focal_available = raw_dataset.shape[1]

    if n_focal_available != focal_size:
        raise ValueError(
            f'{data_file}: expected {focal_size} focal populations but the array holds '
            f'{n_focal_available}.'
        )

    loaded_indices = np.unique(
        np.concatenate([observed_indices, target_indices])
    )

    subset = raw_dataset

    if simulation_indices is not None:
        simulation_indices = np.asarray(simulation_indices, dtype=int)
        subset = subset[simulation_indices]

    subset = subset[:, loaded_indices, :]

    dataset = interpolate_dataset(subset, interp_len=interp_len)
    assert dataset.shape == (subset.shape[0], len(loaded_indices), interp_len)

    n_sim = dataset.shape[0]

    # remap onto positions within the loaded subset
    input_positions = np.searchsorted(loaded_indices, observed_indices)
    target_positions = np.searchsorted(loaded_indices, target_indices)
    assert np.array_equal(loaded_indices[input_positions], observed_indices)
    assert np.array_equal(loaded_indices[target_positions], target_indices)

    # ----------------------------------------------------------------------------------------------------
    # splitting

    # the split unit is per trajectory, so windows from the same trajectory are never split across train and test
    split_indices_name = f'split_indices_seed{split_seed}.json'

    train_indices, test_indices = train_test_split_simulations(
        n_sim=n_sim,
        train_size=train_size,
        split_seed=split_seed,
        output_path=str(result_directory / split_indices_name),
    )

    # ----------------------------------------------------------------------------------------------------
    # forecasting

    # prepare data for regression
    loader = ConsortiaSimDataLoader(
        dataset=dataset,
        train_indices=train_indices,
        test_indices=test_indices,
        input_indices=input_positions,
        target_indices=target_positions,
        input_type=input_type,
        target_type=target_type,
        window_size=window_size,
        stride=stride,
        encoder=encoder,
        z_dim=Z_DIM,
        pca_components=Z_DIM,
        append_max=append_max,
        random_seed=split_seed,
    )

    X_train, y_train, X_test, y_test = loader.get_regression_data()

    logger.info(
        f"    X_train={X_train.shape}, "
        f"    y_train={y_train.shape}"
    )

    logger.info(
        f"    X_test={X_test.shape}, "
        f"    y_test={y_test.shape}"
    )

    # every simulation gives the same number of windows
    windows_per_simulation = X_train.shape[0] // len(train_indices)
    assert X_train.shape[0] == len(train_indices) * windows_per_simulation
    assert X_test.shape[0] == len(test_indices) * windows_per_simulation

    # train regressor
    regressor = ExtraTreesRegressor(
        random_state=split_seed,
        criterion="friedman_mse",
        max_depth=max_depth,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    logger.info("Training ExtraTreesRegressor...")
    regressor.fit(X_train, y_train)

    # ----------------------------------------------------------------------------------------------------
    # performance evaluation

    logger.info("Generating predictions...")
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    # per dim metrics average over (focal population, time index) pairs
    # global metrics flatten everything into a single vector
    r2_train, rmse_train, nrmse_train, global_r2_train, global_rmse_train, global_nrmse_train = compute_metrics(y_train, y_train_pred)
    r2_test, rmse_test, nrmse_test, global_r2_test, global_rmse_test, global_nrmse_test = compute_metrics(y_test, y_test_pred)

    # per focal population R2, only meaningful when the target is the raw curve
    r2_per_focal = None
    if target_type == 'raw':
        n_target_focal = len(target_indices)
        y_test_focal = y_test.reshape(-1, n_target_focal, window_size)
        y_pred_focal = y_test_pred.reshape(-1, n_target_focal, window_size)

        r2_per_focal = [
            float(
                r2_score(
                    y_test_focal[:, focal, :].reshape(-1),
                    y_pred_focal[:, focal, :].reshape(-1),
                )
            )
            for focal in range(n_target_focal)
        ]

    # save results
    results = {
        "timestamp": timestamp,
        "result_directory": str(result_directory),
        "data_file": data_file,
        "simulation": simulation,
        "community_size": community_size,
        "focal_size": focal_size,
        "background_size": community_size - focal_size,

        "n_observed": int(len(observed_indices)),
        "observed_fraction": float(len(observed_indices) / community_size),
        "observed_focal_fraction": float(len(observed_indices) / focal_size),
        "observed_indices": observed_indices.tolist(),
        "target_indices": target_indices.tolist(),
        "n_subsample": int(n_sim),
        "subsample_reference": subsample_reference,

        "input_type": input_type,
        "target_type": target_type,
        "append_max": append_max,
        "interp_len": interp_len,
        "window_size": window_size,
        "stride": stride,
        "train_size": train_size,
        "split_seed": split_seed,
        "split_indices_name": split_indices_name,
        "max_depth": max_depth,
        "n_estimators": n_estimators,

        "n_train_simulations": len(train_indices),
        "n_test_simulations": len(test_indices),
        "windows_per_simulation": int(windows_per_simulation),

        "n_train_windows": int(X_train.shape[0]),
        "n_test_windows": int(X_test.shape[0]),
        "n_input_features": int(X_train.shape[1]),
        "n_target_features": int(y_train.shape[1]),

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

        "r2_per_focal": r2_per_focal,
    }

    results_path = result_directory / 'results.json'

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 80)
    logger.info("Results")
    logger.info(
        f"    Train: R²={r2_train:.4f}, "
        f"    RMSE={rmse_train:.6f}, "
        f"    NRMSE={nrmse_train:.6f}"
    )
    logger.info(
        f"    Test : R²={r2_test:.4f}, "
        f"    RMSE={rmse_test:.6f}, "
        f"    NRMSE={nrmse_test:.6f}"
    )
    logger.info("=" * 80)
    logger.info(f"Saved to: {result_directory}")

    # ----------------------------------------------------------------------------------------------------
    # plotting

    if make_plots:
        train_plots_dir = result_directory / 'train'
        test_plots_dir = result_directory / 'test'
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

        # only the raw target has a time axis to plot metrics against
        if target_type == 'raw':
            try:
                # reshape so the per index axis is time
                plot_per_idx_metrics(
                    y_train.reshape(-1, window_size),
                    y_train_pred.reshape(-1, window_size),
                    title='Train Metrics per Window Internal Index',
                    output_dir=str(train_plots_dir),
                )

                plot_per_idx_metrics(
                    y_test.reshape(-1, window_size),
                    y_test_pred.reshape(-1, window_size),
                    title='Test Metrics per Window Internal Index',
                    output_dir=str(test_plots_dir),
                )
            except Exception as e:
                logger.warning(f'Metrics plotting error: {e}')

    return results


# font sizes for the summary figures
LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 24
TICK_FONTSIZE = 15

XTICK_STEPS = {
    20: 1,
    100: 1,
}

FIGURE_LEFT = 0.10

def plot_summary(df, figure_directory, target_mode):
    """
    Produce summary plots for Figure R4 on the original manuscript data.

    Each figure is a 1x2 panel of global R2 and global RMSE versus the number of observed
    focal populations. Error bars are drawn only when there is more than one split.
    """

    figure_directory = Path(figure_directory)

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # (column prefix, y axis label)
    metrics = [
        ("global_r2_test", "Global R²"),
        ("global_rmse_test", "Global RMSE"),
    ]

    label_map = {
        "raw_to_raw": "Raw to Raw",
        "latent_to_raw": "Latent to Raw",
    }

    for simulation in sorted(df.simulation.unique()):
        for community_size in sorted(df.community_size.unique()):
            subset = df[
                (df.simulation == simulation) & (df.community_size == community_size)
            ]

            if len(subset) == 0:
                continue

            std_columns = [f"{metric}_std" for metric, _ in metrics]
            use_error_bars = bool((subset[std_columns].to_numpy() > 0).any())

            fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
            fig.subplots_adjust(
                left=FIGURE_LEFT,
                right=0.98,
                top=0.86,
                bottom=0.17,
                wspace=0.28,
            )

            for ax, (metric, ylabel) in zip(axes, metrics):
                for representation in sorted(subset.representation.unique()):
                    tmp = subset[subset.representation == representation].sort_values("n_observed")

                    if use_error_bars:
                        ax.errorbar(
                            tmp.n_observed,
                            tmp[f"{metric}_mean"],
                            yerr=tmp[f"{metric}_std"],
                            marker="o",
                            markersize=8,
                            capsize=3,
                            label=label_map[representation],
                        )
                    else:
                        ax.plot(
                            tmp.n_observed,
                            tmp[f"{metric}_mean"],
                            marker="o",
                            label=label_map[representation],
                        )

                ax.set_xlabel("Number of focal populations", fontsize=LABEL_FONTSIZE)
                ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
                ax.tick_params(labelsize=TICK_FONTSIZE)
                ax.grid(True, alpha=0.3)

                max_observed = int(subset.n_observed.max())
                tick_step = XTICK_STEPS.get(community_size, max(1, community_size // 7))

                ax.set_xticks(np.arange(0, max_observed + 1, tick_step))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
                ax.set_xlim(0, max_observed + 0.5)
                
                if metric == "global_r2_test":
                    ax.set_ylim(0.75, 1.01)
                elif metric == "global_rmse_test":
                    if simulation == "bgLV":
                        ax.set_ylim(-0.004, 0.05)
                    elif simulation == "dgLV":
                        ax.set_ylim(0.04, 0.09)


            axes[1].legend(fontsize=TICK_FONTSIZE)

            fig.suptitle(
                f"{simulation} Community with {community_size} Members",
                fontsize=TITLE_FONTSIZE,
                x=FIGURE_LEFT,
                ha="left",
            )

            for ext in [".png", ".pdf"]:
                fig.savefig(
                    figure_directory
                    / f"{simulation}_N{community_size}_{target_mode}{ext}",
                    dpi=300,
                )

            plt.close(fig)


def main(
    data_directory,
    result_directory,
    figure_directory,
    target_mode: str = "single_target", # "single_target" or "multi_target"
    observed_counts=None, # {community_size: (m, ...)}, defaults to OBSERVED_COUNTS
    n_subsample: int = 4000, # simulations kept before the split, None uses every simulation
    n_sim: int = 10000, # simulations stored in each text file
    interp_len: int = SEQ_LEN * 6, # 768, the length the manuscript interpolated to
    window_size: int = SEQ_LEN,
    stride: int = 32, # the legacy step_size, so the windows overlap
    representations=(("raw", "raw"), ("latent", "raw")), # (input_type, target_type) pairs
    append_max: bool = True,
    n_repeats: int = 1,
    base_split_seed: int = 501,
    train_size: float = 0.8,
    max_depth: int = 12,
    n_estimators: int = 25,
    n_jobs: int = None,
    make_plots: bool = True,
):
    """
    Driver for the Figure R4 analysis on the original manuscript datasets.

    Each *_fixed.txt file is swept over how many of its stored focal populations the regressor
    observes, once per representation and once per repeated split, and the metrics are
    aggregated into summary_{target_mode}.csv.
    """

    if target_mode not in ("single_target", "multi_target"):
        raise ValueError(
            f'Unknown target_mode {target_mode}, expected "single_target" or "multi_target".'
        )

    if observed_counts is None:
        observed_counts = OBSERVED_COUNTS

    data_directory = Path(data_directory)
    result_directory = Path(result_directory)
    figure_directory = Path(figure_directory)

    result_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    dataset_files = sorted(data_directory.glob("*_fixed.txt"))

    if len(dataset_files) == 0:
        raise FileNotFoundError(f"No *_fixed.txt files found in {data_directory}")

    # load the encoder once, every latent run reuses it
    needs_encoder = any(
        (input_type == "latent") or (target_type == "latent")
        for input_type, target_type in representations
    )
    encoder = load_default_model() if needs_encoder else None

    # save config
    experiment_config = {
        "data_directory": str(data_directory),
        "dataset_files": [f.name for f in dataset_files],
        "target_mode": target_mode,
        "observed_counts": {str(k): list(v) for k, v in observed_counts.items()},
        "n_subsample": n_subsample,
        "n_sim": n_sim,
        "representations": [
            f"{input_type}_to_{target_type}"
            for input_type, target_type in representations
        ],
        "append_max": append_max,
        "interp_len": interp_len,
        "window_size": window_size,
        "stride": stride,
        "n_repeats": n_repeats,
        "base_split_seed": base_split_seed,
        "train_size": train_size,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
    }

    with open(result_directory / f"experiment_config_{target_mode}.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    records = []

    logger.info("=" * 80)
    logger.info("Focal/background sensitivity analysis, original manuscript data")
    logger.info("=" * 80)

    for dataset_file in dataset_files:

        simulation, community_size, focal_size = parse_dataset_name(dataset_file)

        counts = observed_counts.get(community_size)

        if counts is None:
            logger.warning(
                f"Skipping {dataset_file.name}, observed_counts has no entry for "
                f"community size {community_size}."
            )
            continue

        dataset_directory = result_directory / f"{simulation}_N{community_size}"
        dataset_directory.mkdir(parents=True, exist_ok=True)

        # parsed once per file and held for the whole sweep, at 10000 x 8 x 50 float32 this is 16 MB
        raw_dataset = load_txt_dataset(
            file_path=dataset_file,
            focal_size=focal_size,
            n_sim=n_sim,
        )

        # the subsample is drawn once for this file and then held fixed for every observed count,
        # representation and split below, so the whole sweep is a paired comparison on identical
        # simulations, the seed carries no loop variable so a separate target_mode job reproduces
        # exactly the same subsample
        subsample_reference = None
        simulation_indices = None

        if n_subsample is not None:
            if n_subsample > n_sim:
                raise ValueError(
                    f'n_subsample ({n_subsample}) exceeds the {n_sim} simulations '
                    f'in {dataset_file.name}.'
                )

            subsample_rng = np.random.default_rng(seed=base_split_seed)
            simulation_indices = np.sort(
                subsample_rng.permutation(n_sim)[:n_subsample]
            )

            subsample_reference = 'subsample_indices.json'
            with open(dataset_directory / subsample_reference, 'w') as f:
                json.dump({
                    'base_split_seed': int(base_split_seed),
                    'n_sim_on_disk': int(n_sim),
                    'n_subsample': int(n_subsample),
                    'simulation_indices': simulation_indices.tolist(),
                }, f, indent=2)

        species_permutations = [
            np.random.default_rng(seed=[base_split_seed, fold]).permutation(focal_size)
            for fold in range(n_repeats)
        ]

        for n_observed in counts:

            if n_observed > focal_size:
                raise ValueError(
                    f'Observed count {n_observed} exceeds the {focal_size} focal populations '
                    f'stored in {dataset_file.name}.'
                )

            for input_type, target_type in representations:

                representation_name = f"{input_type}_to_{target_type}"
                logger.info("=" * 80)
                logger.info(f"{dataset_file.name}   m={n_observed}   {representation_name}")
                logger.info("=" * 80)

                split_results = []
                split_seeds = []

                # iterate through repeated trajectory level splits
                for fold in range(n_repeats):

                    permutation = species_permutations[fold]
                    observed_indices = np.sort(permutation[:n_observed])

                    # the anchor is the first population of the permutation
                    if target_mode == "single_target":
                        target_indices = permutation[:1]
                    else:
                        target_indices = observed_indices

                    split_seed = base_split_seed + fold
                    split_seeds.append(split_seed)
                    logger.info(f"    Split {fold + 1}/{n_repeats} (seed={split_seed})")

                    results = run_single_dataset(
                        raw_dataset=raw_dataset,
                        simulation=simulation,
                        community_size=community_size,
                        focal_size=focal_size,
                        result_directory=(
                            dataset_directory
                            / target_mode
                            / f"m{n_observed:03d}"
                            / representation_name
                            / f"split_{fold:02d}"
                        ),
                        observed_indices=observed_indices,
                        target_indices=target_indices,
                        data_file=dataset_file.name,
                        simulation_indices=simulation_indices,
                        subsample_reference=subsample_reference,
                        input_type=input_type,
                        target_type=target_type,
                        interp_len=interp_len,
                        window_size=window_size,
                        stride=stride,
                        append_max=append_max,
                        train_size=train_size,
                        split_seed=split_seed,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        n_jobs=n_jobs,
                        encoder=encoder,
                        make_plots=make_plots,
                    )

                    split_results.append(results)

                record = {
                    "simulation": simulation,
                    "community_size": community_size,
                    "focal_size": focal_size,
                    "background_size": community_size - focal_size,
                    "n_observed": n_observed,
                    "observed_fraction": n_observed / community_size,
                    "observed_focal_fraction": n_observed / focal_size,
                    "n_unobserved": community_size - n_observed,
                    "target_mode": target_mode,
                    "representation": representation_name,
                    "input_type": input_type,
                    "target_type": target_type,
                    "append_max": append_max,
                    "n_repeats": n_repeats,
                    "split_seeds": split_seeds,
                    "n_subsample": split_results[0]["n_subsample"],
                    "n_train_windows": split_results[0]["n_train_windows"],
                    "n_test_windows": split_results[0]["n_test_windows"],
                    "n_input_features": split_results[0]["n_input_features"],
                    "n_target_features": split_results[0]["n_target_features"],
                }

                # append mean and standard deviation over the repeated splits
                for metric in METRIC_NAMES:
                    values = [float(r[metric]) for r in split_results]
                    record[f"{metric}_values"] = values
                    record[f"{metric}_mean"] = float(np.mean(values))
                    record[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

                records.append(record)

    summary_name = f"summary_{target_mode}.csv"

    summary = aggregate_results(
        records=records,
        result_directory=result_directory,
        summary_name=summary_name,
    )

    plot_summary(
        summary,
        figure_directory,
        target_mode=target_mode,
    )

    logger.info("=" * 80)
    logger.info("Finished focal/background sensitivity analysis on the original data.")
    logger.info("=" * 80)
    logger.info(f"Summary CSV : {result_directory / summary_name}")
    logger.info(f"Figures     : {figure_directory}")

    return summary
