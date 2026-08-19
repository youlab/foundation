from pathlib import Path
import json
import csv
import numpy as np

from config import (
    DIR_DATA_KARLSSON,
    DIR_RESULTS,
    SEQ_LEN,
)

from applications.karlsson.forecast import main as forecast_main
from applications.karlsson.karlsson_plots import (
    plot_reconstruction,
    plot_sample_size_results,
    plot_sample_size_summary_figure,
)
from sklearn.metrics import r2_score

FULL_TRAIN_SIZE = 2688
OUTPUT_SUBDIR = "sample_size_analysis_2"

TRAIN_PROPORTIONS = [
    0.0050,
    0.0125,
    0.0625,
    0.1250,
    0.2500,
    0.3750,
    0.5000,
    0.6250,
    0.7500,
    0.8750,
    1.0000,
]

# (input_type, target_type, append_max)
REPRESENTATIONS = [
    ("raw", "raw", False),
    ("pca", "raw", False),
    ("pca", "raw", True),
    ("latent", "raw", False),
    ("latent", "raw", True),
]


# ======================================================================================
# helpers
# ======================================================================================

def _proportions_to_counts():
    counts = []
    for prop in TRAIN_PROPORTIONS:
        n = int(round(prop * FULL_TRAIN_SIZE))
        n = max(1, n)
        counts.append(n)
    return counts


def _fold_seed(base_seed, fold):
    '''deterministic downsampling seeds, returning base_seed + n for 0 <= n < 5'''
    return base_seed + fold


def _representation_name(input_type: str, target_type: str, append_max: bool):
    '''name used for the output subdirectory, the aggregate record and the plot legend'''
    suffix = "_max" if (append_max and input_type != "raw") else ""
    return f"{input_type}{suffix}_to_{target_type}"


def _metric_names():
    return [
        "r2_test",
        "rmse_test",
        "nrmse_test",
        "global_r2_test",
        "global_rmse_test",
        "global_nrmse_test",
    ]


def _summarise(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
    }


def main(
    n_repeats=5,
    random_seed=501,
    base_downsampling_seed=627,
    stride=128,
    max_depth=15,
    plot_only: bool = True,
    plot_start_idx: int = 0,
):

    train_counts = _proportions_to_counts()

    output_dir = Path(DIR_RESULTS) / "karlsson_forecast" / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # test set predictions at the full train set, keyed by representation
    full_train_predictions = {}
    predictions_path = output_dir / "full_train_predictions.npz"

    if plot_only:
        # only run plotting, assuming analysis results are stored
        print("Running plots only")
        json_path = output_dir / "aggregate_results.json"
        with open(json_path, "r") as f:
            aggregate = json.load(f)

        if predictions_path.exists():
            print(f"Loading cached full train set predictions from {predictions_path}")
            cached = np.load(predictions_path)
            for key in cached.files:
                if key.endswith("__y_test"):
                    representation_name = key[: -len("__y_test")]
                    full_train_predictions[representation_name] = (
                        cached[f"{representation_name}__y_test"],
                        cached[f"{representation_name}__y_test_pred"],
                    )
        else:
            print(
                f"[WARNING] {predictions_path.name} not found, "
                "skipping the reconstruction scatters and the summary figure"
            )

    else:
        # save config
        experiment_config = {
            "full_train_size": FULL_TRAIN_SIZE,
            "train_proportions": TRAIN_PROPORTIONS,
            "train_counts": train_counts,
            "representations": [
                _representation_name(input_type, target_type, append_max)
                for input_type, target_type, append_max in REPRESENTATIONS
            ],
            "n_repeats": n_repeats,
            "random_seed": random_seed,
            "base_downsampling_seed": base_downsampling_seed,
            "stride": stride,
            "window_size": SEQ_LEN,
            "max_depth": max_depth,
        }

        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump(experiment_config, f, indent=2)

        aggregate = []

        print("=" * 80)
        print("Sample size analysis")
        print(
            f"{len(REPRESENTATIONS)} representations "
            f"x {len(train_counts)} train sizes "
            f"x {n_repeats} folds "
            f"= {len(REPRESENTATIONS) * len(train_counts) * n_repeats} regressor fits"
        )
        print("=" * 80)

        for input_type, target_type, append_max in REPRESENTATIONS:

            representation_name = _representation_name(input_type, target_type, append_max)
            print()
            print("=" * 80)
            print(representation_name)
            print("=" * 80)

            # iterate through all train downsample proportions
            for proportion, train_count in zip(TRAIN_PROPORTIONS, train_counts):
                print()
                print(
                    f"Training trajectories = {train_count} "
                    f"({100 * proportion:.2f}%)"
                )

                fold_results = []
                fold_seeds = []

                # iterate through n folds
                for fold in range(n_repeats):

                    downsampling_seed = _fold_seed(base_downsampling_seed, fold)
                    fold_seeds.append(downsampling_seed)
                    print(f"    Fold {fold + 1}/{n_repeats} (seed={downsampling_seed})")

                    # run single forecasting task
                    results = forecast_main(
                        data_dir=DIR_DATA_KARLSSON,
                        input_type=input_type,
                        target_type=target_type,
                        append_max=append_max,
                        use_test_indices_at="full_split_indices_seed501.json",
                        split_level="per_replicate",
                        train_size=0.8,
                        train_downsample_to_n=train_count,
                        window_size=SEQ_LEN,
                        stride=stride,
                        max_depth=max_depth,
                        random_seed=random_seed,
                        downsampling_seed=downsampling_seed,
                        output_subdir=(
                            f"{OUTPUT_SUBDIR}"
                            f"/{representation_name}"
                            f"/train_{train_count:04d}"
                            f"/fold_{fold:02d}"
                        ),
                    )

                    fold_results.append(results)

                    if (
                        (train_count == FULL_TRAIN_SIZE)
                        and (fold == 0)
                        and (target_type == "raw")
                    ):
                        full_train_predictions[representation_name] = (
                            np.asarray(results["y_test"], dtype=np.float32),
                            np.asarray(results["y_test_pred"], dtype=np.float32),
                        )

                record = {
                    "representation": representation_name,
                    "input_type": input_type,
                    "target_type": target_type,
                    "append_max": append_max,
                    "train_proportion": proportion,
                    "train_trajectories": train_count,
                    "n_repeats": n_repeats,
                }

                # append results to aggregate list
                for metric in _metric_names():
                    values = [float(r[metric]) for r in fold_results]
                    stats = _summarise(values)
                    record[f"{metric}_values"] = values
                    record[f"{metric}_mean"] = stats["mean"]
                    record[f"{metric}_std"] = stats["std"]
                    record["downsampling_seeds"] = fold_seeds

                aggregate.append(record)

        # save aggregate results
        json_path = output_dir / "aggregate_results.json"
        with open(json_path, "w") as f:
            json.dump(aggregate, f, indent=2)

        with open(json_path, "r") as f:
            aggregate = json.load(f)

        # save aggregate results as csv for plotting
        csv_path = output_dir / "aggregate_results.csv"
        fieldnames = [
            "representation",
            "input_type",
            "target_type",
            "append_max",
            "train_proportion",
            "train_trajectories",
            "n_repeats",
            "downsampling_seeds",
        ]

        for metric in _metric_names():
            fieldnames.extend(
                [
                    f"{metric}_values",
                    f"{metric}_mean",
                    f"{metric}_std",
                ]
            )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in aggregate:
                writer.writerow(row)

        # cache the full train set predictions so plot_only can rebuild the summary figure
        if full_train_predictions:
            cache = {}
            for representation_name, (y_test, y_test_pred) in full_train_predictions.items():
                cache[f"{representation_name}__y_test"] = y_test
                cache[f"{representation_name}__y_test_pred"] = y_test_pred

            np.savez_compressed(predictions_path, **cache)

        print()
        print("=" * 80)
        print("Finished repeated downsampling experiment.")
        print("=" * 80)
        print(f"Aggregate JSON : {json_path}")
        print(f"Aggregate CSV  : {csv_path}")
        print(f"Predictions    : {predictions_path}")


    # plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_sample_size_results(
        aggregate_results=aggregate,
        output_dir=plot_dir,
        start_idx=plot_start_idx,
    )

    if full_train_predictions:

        # one reconstruction scatter per representation at the full train set
        reconstruction_dir = plot_dir / "reconstructions"
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

        for representation_name, (y_test, y_test_pred) in full_train_predictions.items():

            global_r2 = r2_score(
                y_test.reshape(-1),
                y_test_pred.reshape(-1),
            )

            plot_reconstruction(
                y_test,
                y_test_pred,
                title=(
                    f'{representation_name} (full train set): '
                    f'Global R2={global_r2:.3f}'
                ),
                output_path=str(reconstruction_dir / f'{representation_name}.png'),
            )

        plot_sample_size_summary_figure(
            predictions=full_train_predictions,
            aggregate_results=aggregate,
            output_path=plot_dir / "sample_size_summary.png",
            start_idx=plot_start_idx,
        )

    print(f"\nPlots written to {plot_dir}")

    return aggregate



if __name__ == "__main__":

    main(
        n_repeats=5,
        random_seed=501,
        base_downsampling_seed=627,
        stride=128,
        max_depth=15,
        plot_only=False,
    )