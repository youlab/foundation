from pathlib import Path
from datetime import datetime
import json
import logging
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

from applications.consortia.consortia_sim_dataset import (
    load_simulation_dataset,
    train_test_split_simulations,
    ConsortiaSimDataLoader,
)
from applications.consortia.data_generation.focal_background_generation import main as generate_single_dataset
from applications.karlsson.forecast import compute_metrics
from applications.karlsson.karlsson_plots import plot_reconstruction, plot_per_idx_metrics
from ml.utils.load_models import load_default_model

from config import (
    SEQ_LEN,
    Z_DIM,
)

logger = logging.getLogger(__name__)

DATASET_PATTERN = re.compile(
    r"(bgLV|dgLV)_N(\d+)_F(\d+)_I(\d+)_dataset\.npy"
)

# how many focal populations the regressor observes, keyed by community size
OBSERVED_COUNTS = {
    20: (1, 2, 3, 5, 7, 10, 15),
    100: (1, 2, 4, 8, 12, 18, 25),
}

METRIC_NAMES = [
    "r2_test",
    "rmse_test",
    "nrmse_test",
    "global_r2_test",
    "global_rmse_test",
    "global_nrmse_test",
]


def generate_data(
    output_directory,
    bgLV_community_sizes=(20, 100),
    dgLV_community_sizes=(20, 100),
    # focal size equals community size so every species is stored and every initial condition is
    # randomised, which species count as background is then decided at analysis time
    bgLV_focal_sizes=((20,), (100,)),
    dgLV_focal_sizes=((20,), (100,)),
    n_sim=10000,
    model_index=1,
    param_sampling_seed=1,
    bgLV_abundance_threshold=0.10,
    dgLV_abundance_threshold=0.05,
    t_background=294.0,
    t_simulation=300.0,
    dt=6.0,
):
    """
    Generate every dataset required for one focal/background sensitivity run.

    Parameters
    ----------
    bgLV_community_sizes : iterable[int]
        Community sizes to generate for the bounded-gLV model.

    bgLV_focal_sizes : iterable[iterable[int]]
        bgLV_focal_sizes[i] contains the focal sizes corresponding to
        bgLV_community_sizes[i].

    dgLV_community_sizes : iterable[int]
        Community sizes to generate for the dispersal-gLV model.

    dgLV_focal_sizes : iterable[iterable[int]]
        dgLV_focal_sizes[i] contains the focal sizes corresponding to
        dgLV_community_sizes[i].

    Remaining parameters are forwarded directly to
    focal_background_generation.main().
    """

    if len(bgLV_community_sizes) != len(bgLV_focal_sizes):
        raise ValueError(
            "bgLV_community_sizes and bgLV_focal_sizes "
            "must have the same length."
        )

    if len(dgLV_community_sizes) != len(dgLV_focal_sizes):
        raise ValueError(
            "dgLV_community_sizes and dgLV_focal_sizes "
            "must have the same length."
        )

    logger.info("==================================================")
    logger.info("Generating bounded-gLV datasets")
    logger.info("==================================================")

    for community_size, focal_list in zip(
        bgLV_community_sizes,
        bgLV_focal_sizes,
    ):

        for focal_size in focal_list:

            if focal_size <= 0:
                raise ValueError(f"Specified bgLV focal size ({focal_size}) must be at least 1.")
            if focal_size > community_size:
                raise ValueError(f"Requested bgLV focal size ({focal_size}) exceeds community size ({community_size}).")

            logger.info(
                f"Generating bgLV "
                f"(N={community_size}, F={focal_size})"
            )

            generate_single_dataset(
                simulation="bgLV",
                community_size=community_size,
                focal_size=focal_size,
                output_directory=output_directory,
                n_sim=n_sim,
                model_index=model_index,
                param_sampling_seed=param_sampling_seed,
                abundance_threshold=bgLV_abundance_threshold,
                t_background=t_background,
                t_simulation=t_simulation,
                dt=dt,
            )

    logger.info("==================================================")
    logger.info("Generating dispersal-gLV datasets")
    logger.info("==================================================")

    for community_size, focal_list in zip(
        dgLV_community_sizes,
        dgLV_focal_sizes,
    ):

        for focal_size in focal_list:

            if focal_size <= 0:
                raise ValueError(f"Specified dgLV focal size ({focal_size}) must be at least 1.")
            if focal_size > community_size:
                raise ValueError(f"Requested dgLV focal size ({focal_size}) exceeds community size ({community_size}).")

            logger.info(
                f"Generating dgLV "
                f"(N={community_size}, F={focal_size})"
            )

            generate_single_dataset(
                simulation="dgLV",
                community_size=community_size,
                focal_size=focal_size,
                output_directory=output_directory,
                n_sim=n_sim,
                model_index=model_index,
                param_sampling_seed=param_sampling_seed,
                abundance_threshold=dgLV_abundance_threshold,
                t_background=t_background,
                t_simulation=t_simulation,
                dt=dt,
            )

    logger.info("Finished generating all focal/background datasets.")


def parse_dataset_name(dataset_path):
    '''pull the simulation type, community size and focal size out of a dataset filename'''

    match = DATASET_PATTERN.fullmatch(Path(dataset_path).name)

    if match is None:
        raise ValueError(
            f'Unrecognised dataset filename {Path(dataset_path).name}, '
            'expected e.g. bgLV_N20_F5_I1_dataset.npy'
        )

    return (
        match.group(1),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def run_single_dataset(
    dataset_path,
    result_directory,
    observed_indices, # focal populations the regressor observes
    target_indices, # focal populations the regressor has to predict
    simulation_indices=None, # subsample drawn once by main(), never drawn here
    subsample_reference: str = None, # name of the file holding simulation_indices
    input_type: str = 'raw', # 'raw' or 'latent' or 'pca'
    target_type: str = 'raw', # 'raw' or 'latent' or 'pca'
    interp_len: int = SEQ_LEN * 3,
    window_size: int = SEQ_LEN,
    stride: int = SEQ_LEN,
    append_max: bool = True, # append the segment maximum to latent and pca features
    train_size: float = 0.8,
    split_seed: int = 501,
    max_depth: int = 12,
    n_estimators: int = 25,
    n_jobs: int = None,
    encoder=None, # pass a preloaded model to avoid reloading it per run
    make_plots: bool = True,
):
    '''run the forecasting analysis on one simulated dataset for one observed subset and split'''

    dataset_path = Path(dataset_path)
    result_directory = Path(result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    simulation, community_size, focal_size, model_index = parse_dataset_name(dataset_path)

    observed_indices = np.asarray(observed_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)

    logger.info("=" * 80)
    logger.info("Consortia focal/background forecast")
    logger.info(f"    dataset      : {dataset_path.name}")
    logger.info(f"    n_observed   : {len(observed_indices)} of {focal_size}")
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
    # splitting

    # check the filename against the file itself before any subsetting hides the real focal count
    n_focal_on_disk = np.load(dataset_path, mmap_mode='r').shape[1]

    if n_focal_on_disk != focal_size:
        raise ValueError(
            f'{dataset_path.name}: filename specifies {focal_size} focal populations '
            f'but dataset contains {n_focal_on_disk}.'
        )

    # only the observed and target populations are ever touched, loading the rest would make the
    # interpolation scale with the community size instead of with the observed count
    loaded_indices = np.unique(
        np.concatenate([observed_indices, target_indices])
    )

    dataset = load_simulation_dataset(
        file_path=dataset_path,
        interp_len=interp_len,
        focal_indices=loaded_indices,
        simulation_indices=simulation_indices,
    )

    n_sim = dataset.shape[0]

    # remap onto positions within the loaded subset, searchsorted is valid because np.unique sorted
    input_positions = np.searchsorted(loaded_indices, observed_indices)
    target_positions = np.searchsorted(loaded_indices, target_indices)
    assert np.array_equal(loaded_indices[input_positions], observed_indices)
    assert np.array_equal(loaded_indices[target_positions], target_indices)

    # the split unit is one whole simulation, so windows cut below can never straddle the split
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

    # every simulation contributes the same number of windows, this is the oversampling factor
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

    # per dim metrics average over (focal population, time index) pairs, global metrics flatten both
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
        "dataset_file": dataset_path.name,
        "simulation": simulation,
        "community_size": community_size,
        "focal_size": focal_size,
        "model_index": model_index,

        "n_observed": int(len(observed_indices)),
        "observed_fraction": float(len(observed_indices) / community_size),
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
                # reshape so the per index axis is time rather than a (focal, time) pair
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


def aggregate_results(records, result_directory, summary_name='summary.csv'):
    '''collect the per split records into one dataframe of means and standard deviations'''

    result_directory = Path(result_directory)

    df = pd.DataFrame(records)

    df.sort_values(
        [
            "simulation",
            "community_size",
            "representation",
            "n_observed",
        ],
        inplace=True,
    )

    df.to_csv(
        result_directory / summary_name,
        index=False,
    )

    return df


# font sizes for the summary figures
LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 24
TICK_FONTSIZE = 15

# spacing between x ticks for each community size
XTICK_STEPS = {
    20: 3,
    100: 5,
}

FIGURE_LEFT = 0.10

def plot_summary(df, figure_directory, target_mode):
    """
    Produce summary plots for Figure R4.

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

            # at one repeat every standard deviation is zero, an errorbar call would then still
            # draw cap markers into the legend, so fall back to a plain line plot
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

                # the axis stops at the largest observed count actually swept, not at the community
                # size, ticks placed past the data would otherwise drag the view limits out with them
                max_observed = int(subset.n_observed.max())
                tick_step = XTICK_STEPS.get(community_size, max(1, community_size // 7))
                ax.set_xticks(np.arange(0, max_observed + 1, tick_step))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                ax.set_xlim(0, max_observed + 0.5)
                if metric == "global_r2_test":
                    ax.set_ylim(0, 1.01)

            # legend on the right hand panel
            axes[1].legend(fontsize=TICK_FONTSIZE)

            # left aligned with the left edge of the first panel, which is FIGURE_LEFT by construction
            fig.suptitle(
                f"{simulation} Community with {community_size} Members",
                fontsize=TITLE_FONTSIZE,
                x=FIGURE_LEFT,
                ha="left",
            )

            # save a vector copy alongside the raster one for figure assembly
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
    interp_len: int = SEQ_LEN * 3,
    window_size: int = SEQ_LEN,
    stride: int = SEQ_LEN, # stride equal to the window size gives non-overlapping windows
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
    Driver for the complete Figure R4 analysis.

    Every generated *_dataset.npy is swept over the number of observed focal populations,
    once per representation and once per repeated train/test split, and the metrics are
    aggregated into summary.csv.
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

    dataset_files = sorted(data_directory.glob("*_dataset.npy"))

    if len(dataset_files) == 0:
        raise FileNotFoundError(f"No *_dataset.npy files found in {data_directory}")

    # the sweep decides what counts as background, so it needs every species stored, older datasets
    # generated with focal_size < community_size cannot be used here
    usable_files = []
    for dataset_file in dataset_files:
        _, community_size, focal_size, _ = parse_dataset_name(dataset_file)
        if focal_size != community_size:
            logger.warning(
                f"Skipping {dataset_file.name}, it stores {focal_size} of {community_size} "
                "species and the sweep needs all of them."
            )
            continue
        usable_files.append(dataset_file)

    if len(usable_files) == 0:
        raise FileNotFoundError(
            f"No *_dataset.npy files in {data_directory} were generated with focal_size == community_size."
        )

    dataset_files = usable_files

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
    logger.info("Focal/background sensitivity analysis")
    logger.info("=" * 80)

    for dataset_file in dataset_files:

        simulation, community_size, focal_size, model_index = parse_dataset_name(dataset_file)

        counts = observed_counts.get(community_size)

        if counts is None:
            logger.warning(
                f"Skipping {dataset_file.name}, observed_counts has no entry for "
                f"community size {community_size}."
            )
            continue

        dataset_directory = result_directory / f"{simulation}_N{community_size}"
        dataset_directory.mkdir(parents=True, exist_ok=True)

        n_sim_on_disk = np.load(dataset_file, mmap_mode='r').shape[0]

        # the subsample is drawn once for this (simulation, community size) and then held fixed for
        # every observed count, representation and split below, so the whole sweep is a paired
        # comparison on identical simulations, the seed carries no loop variable so a separate
        # target_mode job reproduces exactly the same subsample
        subsample_reference = None
        simulation_indices = None

        if n_subsample is not None:
            if n_subsample > n_sim_on_disk:
                raise ValueError(
                    f'n_subsample ({n_subsample}) exceeds the {n_sim_on_disk} simulations '
                    f'in {dataset_file.name}.'
                )

            subsample_rng = np.random.default_rng(seed=base_split_seed)
            simulation_indices = np.sort(
                subsample_rng.permutation(n_sim_on_disk)[:n_subsample]
            )

            subsample_reference = 'subsample_indices.json'
            with open(dataset_directory / subsample_reference, 'w') as f:
                json.dump({
                    'base_split_seed': int(base_split_seed),
                    'n_sim_on_disk': int(n_sim_on_disk),
                    'n_subsample': int(n_subsample),
                    'simulation_indices': simulation_indices.tolist(),
                }, f, indent=2)

        # one species permutation per repeat, the observed sets are then prefixes of it, so growing
        # the observed count strictly adds information and never swaps the species being predicted
        species_permutations = [
            np.random.default_rng(seed=[base_split_seed, fold]).permutation(community_size)
            for fold in range(n_repeats)
        ]

        for n_observed in counts:

            if n_observed > community_size:
                raise ValueError(
                    f'Observed count {n_observed} exceeds community size {community_size}.'
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

                    # the anchor is the first species of the permutation, so it is observed at every
                    # observed count and single_target predicts the same species across the sweep
                    if target_mode == "single_target":
                        target_indices = permutation[:1]
                    else:
                        target_indices = observed_indices

                    split_seed = base_split_seed + fold
                    split_seeds.append(split_seed)
                    logger.info(f"    Split {fold + 1}/{n_repeats} (seed={split_seed})")

                    results = run_single_dataset(
                        dataset_path=dataset_file,
                        result_directory=(
                            dataset_directory
                            / target_mode
                            / f"m{n_observed:03d}"
                            / representation_name
                            / f"split_{fold:02d}"
                        ),
                        observed_indices=observed_indices,
                        target_indices=target_indices,
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
                    "n_observed": n_observed,
                    "observed_fraction": n_observed / community_size,
                    "n_unobserved": community_size - n_observed,
                    "target_mode": target_mode,
                    "model_index": model_index,
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

    # the two target modes are separate jobs, so keep their summaries in separate files
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
    logger.info("Finished focal/background sensitivity analysis.")
    logger.info("=" * 80)
    logger.info(f"Summary CSV : {result_directory / summary_name}")
    logger.info(f"Figures     : {figure_directory}")

    return summary
