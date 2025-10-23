from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, glob
from matplotlib.ticker import FormatStrFormatter

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from tqdm import tqdm

from applications.antibiotics.loop_for_results import memory_efficient_prediction
from applications.super_resolution.inputs import get_inputs
from config import DIR_RESULTS_SUPER_RESOLUTION
from data.utils import get_data
from ml.utils.load_models import load_default_model

MASK = [85, 90, 91, 92, 93, 94, 95, 96, 97]
INTERVAL = [True, False]
TRAIN_SIZE = [0.1, 1, 10, 100]


def root_mean_squared_error(
    y_true,
    y_pred,
):
    return np.sqrt(
        mean_squared_error(
            y_true=y_true,
            y_pred=y_pred,
        )
    )

# task id extraction
def decode_task_id(task_id):
    mask_idx = task_id // (len(INTERVAL) * len(TRAIN_SIZE))
    interval_idx = (task_id % (len(INTERVAL) * len(TRAIN_SIZE))) // len(TRAIN_SIZE)
    train_idx = task_id % len(TRAIN_SIZE)

    return MASK[mask_idx], INTERVAL[interval_idx], TRAIN_SIZE[train_idx]


def main(
    task_id,
    debug=True,
    n_samples=6,
    max_depth=15,
):
    y_train, y_test = get_data(return_split=True)
    model = load_default_model()

    mask_pct, use_fixed_interval, train_size = list(
        product(
            [85, 90, 91, 92, 93, 94, 95, 96, 97,],
            [True, False,],
            [0.1, 1, 10, 100,],
        )
    )[task_id]

    train_size = int(train_size / 100 * y_train.shape[0])
    y_train = y_train[:train_size]

    results = []
    (
        x_train,
        x_test,
        x_test_interp,
        x_lat_train,
        x_lat_test,
        y_train_full,  # This is now the full-resolution y_train (original space)
    ) = get_inputs(
        y_train=y_train,
        y_test=y_test,
        mask_pct=mask_pct,
        use_fixed_interval=use_fixed_interval,
        model=model,
        debug=debug,
    )
    
    y_test_pred = memory_efficient_prediction(
        x_train=x_train,
        x_test=x_test,
        tgt_train=y_train,
        prediction_model_cache_dir=None,
        name_suffix="test",
        cross_val=-1,
        input_type="raw",
        max_depth=max_depth,
        classify=False,
    )

    # New approach: Use latent features as input, predict directly to original space
    y_lat_test_pred_direct = memory_efficient_prediction(
        x_train=x_lat_train,
        x_test=x_lat_test,
        tgt_train=y_train_full,  # Target is now full-resolution y_train
        prediction_model_cache_dir=None,
        name_suffix="test",
        cross_val=-1,
        input_type="latent_to_full",
        max_depth=max_depth,
        classify=False,
    )
    if debug:
        s = f"MAIN:"
        s += f" x_train {x_train.shape}"
        s += f", x_test {x_test.shape}"
        s += f", x_test_interp {x_test_interp.shape}"
        s += f", x_lat_train {x_lat_train.shape}"
        s += f",\nx_lat_test {x_lat_test.shape}"
        s += f", y_train_full {y_train_full.shape}"
        s += f", y_test_pred {y_test_pred.shape}"
        s += f", y_lat_test_pred_direct {y_lat_test_pred_direct.shape}"
        print(s)

    r2_raw = r2_score(
        y_test.flatten(),
        y_test_pred.flatten(),
    )
    r2_lat_direct = r2_score(
        y_test.flatten(),
        y_lat_test_pred_direct.flatten(),
    )
    r2_naive = r2_score(
        y_test.flatten(),
        x_test_interp.flatten(),
    )
    rmse_raw = root_mean_squared_error(
        y_test.flatten(),
        y_test_pred.flatten(),
    )
    rmse_lat_direct = root_mean_squared_error(
        y_test.flatten(),
        y_lat_test_pred_direct.flatten(),
    )
    rmse_naive = root_mean_squared_error(
        y_test.flatten(),
        x_test_interp.flatten(),
    )
    results.append(
        {
            "mask_pct": mask_pct,
            "use_fixed_interval": use_fixed_interval,
            "train_size": train_size,
            "max_depth": max_depth,
            "train_size": y_train.shape[0],
            "test_size": y_test.shape[0],
            "n_points": x_test.shape[1],
            "r2_naive": r2_naive,
            "r2_raw": r2_raw,
            "r2_lat_direct": r2_lat_direct,
            "rmse_naive": rmse_naive,
            "rmse_raw": rmse_raw,
            "rmse_lat_direct": rmse_lat_direct,
        }
    )

    pd.DataFrame.from_records(results).to_csv(
        DIR_RESULTS_SUPER_RESOLUTION
        / f"results_{task_id}.csv",
    )

    np.savez(
        DIR_RESULTS_SUPER_RESOLUTION
        / f"samples_{task_id}.npz",
        y_test=y_test[:n_samples, :],
        x_test=x_test[:n_samples, :],
        x_test_interp=x_test_interp[:n_samples, :],
        y_test_pred=y_test_pred[:n_samples, :],
        y_lat_test_pred_direct=y_lat_test_pred_direct[:n_samples, :],
    )


if __name__ == "__main__":

    # fitting extra trees
    '''
    tasks = list(product(MASK, INTERVAL, TRAIN_SIZE))
    for task_id in range(len(tasks)):
        main(task_id=task_id, debug=False)
    '''

    # plotting
    csvs = glob.glob(str(DIR_RESULTS_SUPER_RESOLUTION / 'results_*.csv'))
    df_all = pd.concat((pd.read_csv(f) for f in csvs), ignore_index=True)

    mask_percentages = sorted(df_all['mask_pct'].unique())

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), sharey=True, sharex=True)
    axes = axes.flatten()

    for i, mask in enumerate(mask_percentages):
        ax = axes[i]
        plot_settings = [
            ("rmse_naive", True, "Linear interpolation fixed","tab:blue", "-"),
            ("rmse_raw", True, "Raw prediction fixed", "tab:orange", "-"),
            ("rmse_lat_direct", True, "Latent prediction fixed", "tab:green", "-"),
            ("rmse_naive", False, "Linear interpolation random","tab:red", "-"),
            ("rmse_raw", False, "Raw prediction random", "tab:purple", "-"),
            ("rmse_lat_direct", False, "Latent prediction random","tab:brown","-")
        ]

        for metric, fixed, label, color, linestyle in plot_settings:
            subdf = df_all[(df_all["mask_pct"] == mask) & (df_all["use_fixed_interval"] == fixed)]
            subdf = subdf.sort_values("train_size")
            ax.plot(subdf["train_size"], subdf[metric], marker="o", label=label, color=color, linestyle=linestyle)

        ax.set_xscale("log")
        ax.set_title(f"mask={mask}%", fontsize=14, loc="right")

        if i // 3 == 2:
            ax.set_xlabel("Train size", fontsize=12)
        if i % 3 == 0:
            ax.set_ylabel("RMSE", fontsize=12)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    outpath = DIR_RESULTS_SUPER_RESOLUTION / "rmse_plot.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


    # plot R2 result
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), sharey=True, sharex=True)
    axes = axes.flatten()

    for i, mask in enumerate(mask_percentages):
        ax = axes[i]
        plot_settings = [
            ("r2_naive", True, "Linear interpolation fixed", "tab:blue", "-"),
            ("r2_raw", True, "Raw prediction fixed", "tab:orange",   "-"),
            ("r2_lat_direct", True, "Latent prediction fixed", "tab:green", "-"),
            ("r2_naive", False, "Linear interpolation random", "tab:red", "-"),
            ("r2_raw", False, "Raw prediction random", "tab:purple", "-"),
            ("r2_lat_direct", False, "Latent prediction random","tab:brown", "-")
        ]

        for metric, fixed, label, color, linestyle in plot_settings:
            subdf = df_all[(df_all["mask_pct"] == mask) & (df_all["use_fixed_interval"] == fixed)]
            subdf = subdf.sort_values("train_size")
            ax.plot(subdf["train_size"], subdf[metric], marker="o", label=label, color=color, linestyle=linestyle)

        ax.set_xscale("log")
        ax.set_title(f"mask={mask}%", fontsize=14, loc="right")

        if i // 3 == 2:
            ax.set_xlabel("Train size", fontsize=12)
        if i % 3 == 0:
            ax.set_ylabel("R²", fontsize=12)
            
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    outpath = DIR_RESULTS_SUPER_RESOLUTION / "r2_plot.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    # per sample plots
    def plot_samples(task_id):
        samples = np.load(DIR_RESULTS_SUPER_RESOLUTION / f"samples_{task_id}.npz")

        y_test = samples["y_test"]
        x_test = samples["x_test"]
        x_test_interp = samples["x_test_interp"]
        y_test_pred = samples["y_test_pred"]
        y_lat_test_pred_direct = samples["y_lat_test_pred_direct"]

        n_samples = y_test.shape[0]

        mask_pct, fixed_flag, train_frac = decode_task_id(task_id)
        fixed = "fixed interval" if fixed_flag else "random interval"
        train_size = int(train_frac / 100 * get_data(return_split=True)[0].shape[0])

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
        axes = axes.flatten()

        for i in range(min(n_samples, 6)):
            ax = axes[i]

            time_idx = np.arange(y_test.shape[1])
            input_idx = np.linspace(0, y_test.shape[1]-1, x_test.shape[1], dtype=int)

            # true + inputs
            ax.plot(time_idx, y_test[i], label="True", color="blue")
            ax.plot(input_idx, x_test[i], "o", label="Input", color="blue")

            # interpolation
            rmse_interp = root_mean_squared_error(y_test[i].flatten(), x_test_interp[i].flatten())
            ax.plot(time_idx, x_test_interp[i], label=f"Lin int, RMSE={rmse_interp:.3f}", color="orange")

            # raw pred
            rmse_raw = root_mean_squared_error(y_test[i].flatten(), y_test_pred[i].flatten())
            ax.plot(time_idx, y_test_pred[i], label=f"Raw pred, RMSE={rmse_raw:.3f}", color="green")

            # latent pred
            rmse_lat = root_mean_squared_error(y_test[i].flatten(), y_lat_test_pred_direct[i].flatten())
            ax.plot(time_idx, y_lat_test_pred_direct[i], label=f"Latent pred, RMSE={rmse_lat:.3f}", color="red")

            ax.set_xlabel("Time")
            ax.set_ylabel("RMSE")
            ax.legend(fontsize=8, loc="lower right", framealpha=0.8)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)


        fig.suptitle(f"Mask = {mask_pct}%, {fixed}, train size = {train_size}", fontsize=14, ha="center")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        outpath = DIR_RESULTS_SUPER_RESOLUTION / f"sample_plots/sample_plot_{task_id}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()


    for task_id in range(72):
        plot_samples(task_id)