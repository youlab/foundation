import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS_MODEL,
    DIR_RESULTS_MODEL_COMPARISON,
    MODEL_TYPE,
    Z_DIM,
)


def plot_results(
    df_mean,
    df_se,
):
    FS_LABELS = 16
    FS_TICKS = 14
    
    plt.ion()
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(
            8, 4,
        ),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    ax[0].plot(
        df_mean.intrinsic_dimensions_int.to_numpy(),
        df_mean.r2_train_all.to_numpy(),
        marker="o",
        alpha=0.8,
        color="tab:blue",
        label="training mean",
    )
    ax[0].fill_between(
        df_mean.intrinsic_dimensions_int.to_numpy(),
        df_mean.r2_train_all.to_numpy() - df_se.r2_train_all.to_numpy(),
        df_mean.r2_train_all.to_numpy() + df_se.r2_train_all.to_numpy(),
        alpha=0.4,
        label="training standard error",
    )
    ax[1].plot(
        df_mean.intrinsic_dimensions_int.to_numpy(),
        df_mean.r2_test_all.to_numpy(),
        marker="o",
        alpha=0.8,
        color="tab:orange",
        label="test mean",
    )
    ax[1].fill_between(
        df_mean.intrinsic_dimensions_int.to_numpy(),
        df_mean.r2_test_all.to_numpy() - df_se.r2_test_all.to_numpy(),
        df_mean.r2_test_all.to_numpy() + df_se.r2_test_all.to_numpy(),
        alpha=0.4,
        color="tab:orange",
        label="test standard error",
    )
    for i in range(2):
        ax[i].set_xticks(
            [0, 5, 10, 15, 20,],
            labels=[0, 5, 10, 15, 20,],
            fontsize=FS_TICKS,
        )
        for spine in ["top", "right",]:
            ax[i].spines[spine].set_visible(False)
    fig.supylabel("Reconstruction Accuracy (" + r"$R^2$" + ")", fontsize=FS_LABELS,)
    fig.supxlabel("Intrinsic Dimensions", fontsize=FS_LABELS,)
    ax[0].set_yticks(
        [0.9, 0.95, 1.0,],
        labels=[0.90, 0.95, 1.0],
        fontsize=FS_TICKS,
    )
    ax[0].legend(loc="lower left",)
    ax[1].legend(loc="lower left",)
    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_3.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_3.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_3.svg",
    )
    plt.close()


def main():
    print("Plotting intrinsic dimensions")
    df_dim = pd.read_csv(
        DIR_RESULTS_MODEL
        / "intrinsic_dimensions.csv",
        index_col=0,
    )
    df_dim = df_dim.loc[df_dim.intrinsic_dimensions > 0]
    
    df_dim["intrinsic_dimensions_int"] = np.around(df_dim.intrinsic_dimensions).astype(int)
    df_recon = pd.read_csv(
        DIR_RESULTS_MODEL_COMPARISON
        / f"recon_by_dataset_{MODEL_TYPE}_{Z_DIM}.csv",
    index_col=0,)
    df = df_recon.merge(df_dim, on="file_name", how="inner",)
    df_mean = df.groupby("intrinsic_dimensions_int", as_index=False,).agg("mean",numeric_only=True)
    df_se = df.select_dtypes(include=[np.number]).groupby(df["intrinsic_dimensions_int"], as_index=False,).agg(lambda x: x.std(ddof=1) / np.sqrt(x.count()))

    plot_results(
        df_mean=df_mean,
        df_se=df_se,
    )
