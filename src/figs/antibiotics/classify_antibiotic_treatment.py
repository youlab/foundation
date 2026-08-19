import matplotlib.pyplot as plt
import pandas as pd

from config import (
    DIR_RESULTS_ANTIBIOTICS
)


def get_data(model_name):
    return pd.read_csv(
        DIR_RESULTS_ANTIBIOTICS
        / f"classify_antibiotic_summary_{model_name}.csv",
        index_col=0,
    )


def main(
    model_name,
    metric_labels=[
        "raw_accuracy",
        "latent_accuracy",
    ],
    ax=None,
    add_formatting=True,
    add_legend=False,
    fs_labels=18,
    fs_legend=14,
    fs_ticks=14,
    fs_title=20,
):
    
    legend_key = {
        "raw_accuracy": "Using raw",
        "latent_accuracy": "Using latent",
    }

    data = get_data(model_name=model_name)

    ax_existed = not ax is None
    if not ax_existed:
        plt.ion()
        _, ax = plt.subplots(
            1,
            1,
            figsize=(
                8,
                8,
            ),
            constrained_layout=True,
            sharex=False,
            sharey=False,
        )
        ax_existed = False

    # for label in metric_labels:
    #     ax.plot(
    #         data.train_size.to_numpy(),
    #         data.loc[:, label].to_numpy(),
    #         label=legend_key[label],
    #         marker="o",
    #     )

    std_key = {
        "raw_accuracy": "raw_accuracy_std",
        "latent_accuracy": "latent_accuracy_std",
    }

    for label in metric_labels:
        ax.errorbar(
            data.train_size.to_numpy(),
            data[label].to_numpy(),
            yerr=data[std_key[label]].to_numpy(),
            label=legend_key[label],
            marker="o",
            markersize=5,
            capsize=3,
            elinewidth=1,
        )

    if add_legend:
        ax.legend(
            fontsize=fs_legend,
            loc="lower right",
        )
    if add_formatting:
        if not ax_existed:
            ax.set_title(f"Classification Accuracy", fontsize=fs_title,)
        ax.set_ylabel("Classification R²", fontsize=fs_labels,)
        ax.set_xlabel("Training Curves", fontsize=fs_labels,)
        ax.set_xticks(
            [0, 2000, 4000,],
            labels=[0, 2000, 4000,],
            fontsize=fs_ticks,
        )
        ax.set_yticks(
            [0.2, 0.4, 0.6, 0.8, 1.0,],
            labels=[0.2, 0.4, 0.6, 0.8, 1.0,],
            fontsize=fs_ticks,
        )
    for spine in ["top", "right",]:
        ax.spines[spine].set_visible(False)
    ax.text(
        100,
        0.95,
        "using latent",
        color="tab:orange",
        fontsize=14,
        fontweight="medium",
        verticalalignment="top",
    )
    ax.text(
        2000,
        0.75,
        "using raw",
        color="tab:blue",
        fontsize=14,
        fontweight="medium",
    )
    if not ax_existed:
        plt.savefig(
            DIR_RESULTS_ANTIBIOTICS
            / f"classify_antibiotic_treatment_{model_name}.png",
        )

        plt.close()
