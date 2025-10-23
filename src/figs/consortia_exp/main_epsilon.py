import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from applications.config import NOW_TEXT
from applications.utils.metrics import get_rmse
from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_CACHE_CONSORTIA_EXP,
    SEQ_LEN,
)
from figs.consortia.pipeline import plot_pipeline
from figs.consortia_exp.experiment_conditions import plot_experiment_conditions
from figs.consortia_exp.plot_abundance_classification import main as plot_abundance_classification
from figs.consortia_exp.plot_forecast_rmse import main as plot_forecast_rmse
from figs.utils.accuracy_over_time import plot_rmse
from figs.utils.colors import get_colors
from figs.utils.cross_validation import plot_cross_validation
from figs.utils.pipelines import plot_classification_pipeline
from figs.utils.trajectory import plot_example

FS_LABELS = 20
FS_TICKS = 18


def plot_cv(
    ax,
    fs_number=10,
    fs_text=14,
    w_train=60,
    w_vert=20,    
):
    plot_cross_validation(
        ax=ax,
        left=0,
        w=0.05,
        h=0.1,
        x_space=0.03,
        y_space=0.05,
        y0=0.8,
        rmse_right_adder=0.9,
        w_train=w_train,
        w_vert=w_vert,
        marker_size=3,
        h_dots=0.07,
        space_dots=0.05,
        fs_number=fs_number,
        fs_text=fs_text,
        length_a=10,
    )
    
    ax.set_xlim(-0.2, 1.1,)
    ax.set_ylim(0.15, 1.1)
    ax.axis("off")


def get_y_true_and_y_pred(
    experiment="water-b",
):
    if experiment == "water-b":
        reps = ["1", "2", "3a", "3b", "4", "5", "6", "7",]
    else:
        reps = ["1", "2", "3", "4", "5", "6", "7", "8",]
    
    y_true = {}
    y_pred = {}
    y_true_for_rmse = []
    y_pred_for_rmse = []
    y_true_for_rmse_raw = []
    y_pred_for_rmse_raw = []
    for rep in reps:
        data = np.load(
            DIR_CACHE_CONSORTIA_EXP
            / f"future_outlook_{experiment}-{rep}_latent_latent_{NOW_TEXT}.npz",
        )
        y_true_for_rmse.append(data["w_test"])
        y_pred_for_rmse.append(data["w_test_pred"])
        data_raw = np.load(
            DIR_CACHE_CONSORTIA_EXP
            / f"future_outlook_{experiment}-{rep}_raw_segment128_{NOW_TEXT}.npz",
        )
        y_true_for_rmse_raw.append(data_raw["w_test"])
        y_pred_for_rmse_raw.append(data_raw["w_test_pred"])
        for i in range(5):
            i1 = SEQ_LEN * (i + 1)
            i2 = i1 + SEQ_LEN

            if i not in y_true:
                y_true[i] = []
                y_pred[i] = []
            y_true[i].append(
                data["w_test"][:, :, i1:i2].flatten()
            )
            y_pred[i].append(
                data["w_test_pred"][:, :, i1:i2].flatten()
            )
    
    for key in y_true.keys():
        y_true[key] = np.concatenate(y_true[key])
        y_pred[key] = np.concatenate(y_pred[key])
    y_true_for_rmse = np.concatenate(y_true_for_rmse)
    y_pred_for_rmse = np.concatenate(y_pred_for_rmse)
    y_true_for_rmse_raw = np.concatenate(y_true_for_rmse_raw)
    y_pred_for_rmse_raw = np.concatenate(y_pred_for_rmse_raw)
    _, rmse, _ = get_rmse(
        y_true=y_true_for_rmse,
        y_pred=y_pred_for_rmse,
    )
    _, rmse_raw, _ = get_rmse(
        y_true=y_true_for_rmse_raw,
        y_pred=y_pred_for_rmse_raw,
    )
    return y_true, y_pred, rmse, rmse_raw


def get_axes():
    fig = plt.figure(
        figsize=(16, 9,),
        layout="tight",
    )
    return fig, fig.subplot_mosaic(
        "A;D;F",
        height_ratios=[2,2.7,2,],
    )


def plot_forecasts(
    ax,
    data,
    experiment,
    p_y=0.02,
    p_x=0.01,
    side_legend=False,
    fs_legend=14,
    n_col=4,
    n_panels=12,
):
    _, _, rmse, rmse_raw = get_y_true_and_y_pred(experiment=experiment)

    colors = get_colors(n=n_panels - 1)
    n_row = n_panels // n_col
    x_step = 1 / n_col
    y_step = 1 / n_row
    for i in range(n_panels):
        row = n_row - (i // n_col + 1)
        col = i % n_col
        if i != (n_panels - 1):
            a = ax.inset_axes(
                [col * x_step + p_x, row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
            )
            plot_example(
                a=a,
                data=(
                    data["w_test"][0][i:i+1],
                    data["w_test_pred"][0][i:i+1],
                ),
                add_ylabel=False,
                lw_true=5,
                lw_pred=2,
                colors=[colors[i]],
            )

            for j in range(7):
                a.plot(
                    [127 + SEQ_LEN * j, 127 + SEQ_LEN * j,],
                    [0, 1,] if i != 11 else [0, 0.3],
                    "k:",
                    alpha=0.6,
                )
                if (row == 2) and (col == 0):
                    start = (90 + SEQ_LEN * j, 1.0)
                    end = (170 + SEQ_LEN * j, 1.0)
                    
                    arrow = FancyArrowPatch(
                        start, end, 
                        connectionstyle="arc3,rad=-0.4",  # Controls curvature
                        mutation_scale=7,  # Size of arrowhead
                        color="black",
                    )
                    a.add_patch(arrow)

            if (row == 0) and (col == 0):
                a.set_yticks(
                    [0, 0.5, 1,],
                    labels=[0, "", 1],
                    fontsize=FS_TICKS,
                )
                a.set_xticks(
                    [0, 512, 1024,],
                    labels=[0, 55, 110,],
                    fontsize=FS_TICKS,
                )
                a.set_ylabel(
                    "Cell density",
                    fontsize=FS_LABELS,
                )
                a.set_xlabel(
                    "Time (days)",
                    fontsize=FS_LABELS,
                )
            else:
                a.set_yticks(
                    [0, 0.5, 1,],
                    labels=[],
                )
                a.set_xticks(
                    [0, 512, 1024,],
                    labels=[],
                )
            a.set_ylim(0, 1.1)
            
        else:
            p_x *= 2
            p_y *= 2
            a = ax.inset_axes(
                [col * x_step + p_x, row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
            )    
            plot_rmse(
                a=a,
                data=rmse[128:],
                color="cornflowerblue",
                ls="-",
                lw=2,
            )

            plot_rmse(
                a=a,
                data=rmse_raw[128:],
                color="mediumorchid",
                ls=":",
                lw=2,
            )
            for j in range(7):
                a.plot(
                    [127 + SEQ_LEN * j, 127 + SEQ_LEN * j,],
                    [0, 0.3],
                    "k:",
                    alpha=0.6,
                )

            a.set_yticks(
                [0, 0.15, 0.3,],
                labels=[0, "", 0.3,],
                fontsize=FS_TICKS,
            )
            a.set_xticks(
                [0, 512, 1024,],
                labels=[0, 55, 110,],
                fontsize=FS_TICKS,
            )
            if side_legend:
                a.legend(
                    bbox_to_anchor=(1.05, 1,),
                    loc="upper left",
                    fontsize=fs_legend,
                )
            else:
                a.legend(
                    loc="lower right",
                    ncols=2,
                )
            a.set_ylim(0, 0.3,)
        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)
        a.set_xlim(0, 1_024)
    ax.axis("off")


def main(
    fs_panel,
    fw_panel,
):
    fig, ax_dict = get_axes()

    experiment = "water-a"
    rep = "3"
    data = np.load(
        DIR_CACHE_CONSORTIA_EXP
        / f"future_outlook_{experiment}-{rep}_latent_latent_{NOW_TEXT}.npz",
    )

    ax = []
    for bounds in [
        [0.04, 0, 0.15, 1,],
        [0.3, 0.1, 0.25, 0.8,],
        [0.6, 0.1, 0.4, 0.8,],
    ]:
        ax.append(
            ax_dict["A"].inset_axes(
                bounds,
            )
        )

    plot_experiment_conditions(a=ax[0])
    ax_dict["A"].axis("off")

    plot_cv(
        ax=ax[1],
        w_train=55,
        w_vert=40,
    )

    plot_pipeline(
        a=ax[2],
        data=data,
        alpha=0.6,
        y_label=1.1,
    )

    ax = ax_dict["D"].inset_axes(
        [0.03, 0.05, 0.95, 0.95,],
    )
    ax_dict["D"].axis("off")
    plot_forecasts(
        ax=ax,
        data=data,
        experiment=experiment,
    )
    
    ax = []
    for bounds in [
        [0.1, 0.1, 0.18, 0.8,],
        [0.3, 0.1, 0.35, 0.8,],
        [0.8, 0.1, 0.18, 0.8,],
    ]:
        ax.append(
            ax_dict["F"].inset_axes(
                bounds,
            )
        )

    plot_forecast_rmse(
        ax=ax[0],
    )

    plot_classification_pipeline(
        a=ax[1],
        data=data,
        w=40,
    )
    plot_abundance_classification(
        ax=ax[2],
    )
    ax_dict["F"].axis("off")

    for x, y, label in [
        (0.02, 0.97, "A",),
        (0.27, 0.97, "B",),
        (0.59, 0.97, "C",),
        (0.02, 0.7, "D",),
        (0.02, 0.3, "E",),
        (0.28, 0.3, "F",),
        (0.69, 0.3, "G",),
    ]:
        fig.text(
            x=x,
            y=y,
            s=label,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=fs_panel,
            fontweight=fw_panel,
        )


    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_5.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_5.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_5.svg",
    )
    plt.close()
