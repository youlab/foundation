import matplotlib.pyplot as plt
import numpy as np

from applications.config import NOW_TEXT
from applications.consortia.config import get_consortia_for_plotting
from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_CACHE_CONSORTIA,
    MODEL_TYPE,
    SEQ_LEN,
    Z_DIM,
)
from figs.consortia.data import load_data
from figs.consortia.sliding_window import show_sliding_window
from figs.utils.accuracy_over_time import (
    plot_r2,
    plot_rmse,
)
from figs.utils.colors import get_colors
from figs.utils.symbols import (
    get_vae_patches,
    add_arrow,
    add_regr_rectangle,
)
from figs.utils.trajectory import plot_example

FS_LABELS = 20
FS_TICKS = 18


def get_axes():
    fig = plt.figure(
        figsize=(16, 9,),
        layout="tight",
    )
    ax_dict = fig.subplot_mosaic(
        "A;B;D",
        height_ratios=[2,2,3,],
    )

    ax_off = {
        "A": {
            "x_lim": (0, 16,),
        },
    }
    x_ticks = [0, 128, 256, 384, 512, 640, 768,]
    x_vals = {
        "B": {
            "ticks": x_ticks,
            "tick_labels": [],
        },
        "D": {
            "ticks": x_ticks,
            "tick_labels": [],
        },
    }
    y_vals = {}
    for ax_label in ax_dict.keys():
        if ax_label in ax_off:
            if "x_lim" in ax_off[ax_label]:
                ax_dict[ax_label].set_xlim(ax_off[ax_label]["x_lim"])
            continue
        if ax_label in x_vals:
            if "label" in x_vals[ax_label]:
                ax_dict[ax_label].set_xlabel(
                    x_vals[ax_label]["label"],
                    fontsize=FS_LABELS,
                )
            ax_dict[ax_label].set_xticks(
                x_vals[ax_label]["ticks"],
                labels=x_vals[ax_label]["tick_labels"],
                fontsize=FS_TICKS,
            )
            ax_dict[ax_label].set_xlim(
                x_vals[ax_label]["ticks"][0],
                x_vals[ax_label]["ticks"][-1],
            )
        else:
            ax_dict[ax_label].set_xticks([])
        if ax_label in y_vals:
            if "label" in y_vals[ax_label]:
                ax_dict[ax_label].set_ylabel(
                    y_vals[ax_label]["label"],
                    fontsize=FS_LABELS,
                )
            ax_dict[ax_label].set_yticks(
                y_vals[ax_label]["ticks"],
                labels=y_vals[ax_label]["tick_labels"],
                fontsize=FS_TICKS,
            )
            ax_dict[ax_label].set_ylim(
                y_vals[ax_label]["ticks"][0],
                y_vals[ax_label]["ticks"][-1],
            )
        else:
            ax_dict[ax_label].set_yticks([])
        for spine in ["right", "top",]:
            ax_dict[ax_label].spines[spine].set_visible(False)

    ax_dict["A"].axis("off")

    return fig, ax_dict


def plot_pipeline(
    a,
    data,
    alpha=0.8,
    y_label=0.9,
    fs_text=16,
    lw=4,
):
    n_members =  data["w_test"].shape[1]
    colors = get_colors(
        n=n_members,
    )
    idx = SEQ_LEN * 2
    SPACE = 0.15
    
    for i_member in range(n_members):
        a.plot(
            np.linspace(0, 1, SEQ_LEN),
            data["w_test"][0, i_member, idx:idx + SEQ_LEN],
            color=colors[i_member],
            lw=lw,
            alpha=alpha,
        )

    a.text(
        x=0.5,
        y=y_label,
        s="Input",
        fontsize=fs_text,
        horizontalalignment="center",
        verticalalignment="center",
    )

    vae_w_total = 0.9
    right = 1
    right = get_vae_patches(
        a=a,
        left=right + SPACE,
        just_encoder_to_latent=True,
        w_total=vae_w_total,
    )

    right = add_arrow(
        a=a,
        x=right + SPACE,
        y=0.5,
        dx=0.3,
        dy=0,
    )

    right = add_regr_rectangle(
        a=a,
        xy=(right + SPACE, 0.25,),
        width=0.2,
        height=0.5,
    )

    right = add_arrow(
        a=a,
        x=right + SPACE,
        y=0.5,
        dx=0.3,
        dy=0,
    )

    right = get_vae_patches(
        a=a,
        left=right + SPACE,
        just_latent_to_decoder=True,
        w_total=vae_w_total,
    )

    print('data["w_test"].shape)', data["w_test"].shape)
    print(n_members)
    left = right + SPACE
    for i_member in range(n_members):
        a.plot(
            np.linspace(left, left + 1, SEQ_LEN),
            data["w_test"][0, i_member, idx + SEQ_LEN:idx + SEQ_LEN * 2],
            color=colors[i_member],
            lw=lw,
            alpha=alpha,
        )

    a.text(
        x=left + 0.5,
        y=y_label,
        s="Predicted",
        fontsize=fs_text,
        horizontalalignment="center",
        verticalalignment="center",
    )

    a.axis("off")


def plot_labels_on_plot(
    ax,
    add_label_to_plot,
):
    x, y, x_pad, y_pad = add_label_to_plot
    len_line = 64
    ax.plot(
        [x, x + len_line,],
        [y, y,],
        lw=5,
        color="k",
        alpha=0.4,
    )
    ax.plot(
        [x, x + len_line,],
        [y - y_pad, y - y_pad,],
        lw=2,
        color="k",
    )
    ax.text(
        x=x+len_line+x_pad,
        y=y,
        s="true",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=12,
    )
    ax.text(
        x=x+len_line+x_pad,
        y=y-y_pad,
        s="latent forecast",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=12,
    )


def plot_forecasts(
    ax,
    df,
    consortium,
    p_y=0.02,
    p_x=0.01,
    side_legend=False,
    legend_loc="upper left",
    fs_legend=14,
    fs_accuracy=16,
    legend_cols=1,
    train_size="8000",
    accuracy_plot_right_pad=4,
    add_label_to_plot=None,
):
    n_members = { 
        "simple": 5,
        "complex": 8,
    }
    colors = get_colors(n=n_members[consortium])
    mask_consortium = df.consortium == consortium
    mask_train_size = mask_consortium & (df.train_size == train_size)
    input_type = "latent"
    mask = mask_train_size & (df.input_type == input_type)
    print(mask_consortium.sum(), mask.sum(), mask_train_size.sum())
    data = df.loc[mask, "test_example"].iloc[0]
    n_col = 3
    n_row = (n_members[consortium] + 1) // n_col
    x_step = 1 / n_col
    y_step = 1 / n_row
    y_true, y_pred = data
    for i in range(n_members[consortium] + 1):
        row = n_row - (i // n_col + 1)
        col = i % n_col
        if i != n_members[consortium]:
            a = ax.inset_axes(
                [col * x_step + p_x, row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
            )
            if (i == 0) and (add_label_to_plot is not None):
                plot_labels_on_plot(
                    ax=a,
                    add_label_to_plot=add_label_to_plot,
                )
            plot_example(
                a=a,
                data=(
                    y_true[i:i+1],
                    y_pred[i:i+1],
                ),
                add_ylabel=False,
                lw_true=5,
                lw_pred=2,
                colors=[colors[i]],
            )
            a.plot(
                [SEQ_LEN - 1, SEQ_LEN - 1,],
                [0, 1,],
                "k:",
            )

        else:
            a = ax.inset_axes(
                [col * x_step + p_x * 2 + p_x * 6, row * y_step + p_y, x_step - 2 * p_x * 2 * accuracy_plot_right_pad - 4 * p_x, y_step - 2 * p_y,]
            )
            input_type = "latent"
            mask = mask_train_size & (df.input_type == input_type)
            data = df.loc[mask, "test_example"].iloc[0]
            r2_ = df.loc[mask_train_size & (df.input_type == "latent"), "r2_test"].iloc[0].min()
            rmse_ = df.loc[mask_train_size & (df.input_type == "latent"), "rmse_test"].iloc[0].max()
            print(f"latent R2 min {r2_}, RMSE max {rmse_}")
            # r2_ = df.loc[mask_train_size & (df.input_type == "raw"), "r2_test"].iloc[0].min()
            # rmse_ = df.loc[mask_train_size & (df.input_type == "raw"), "rmse_test"].iloc[0].max()
            # print(f"raw R2 min {r2_}, RMSE max {rmse_}")
            plot_r2(
                a=a,
                data=df.loc[mask_train_size & (df.input_type == "latent"), "r2_test"].iloc[0],
                color="firebrick",
                ls="-",
                lw=2,
            )
            # plot_r2(
            #     a=a,
            #     data=df.loc[mask_train_size & (df.input_type == "raw"), "r2_test"].iloc[0],
            #     color="firebrick",
            #     ls=":",
            #     lw=2,
            # )
            plot_rmse(
                a=a,
                data=df.loc[mask_train_size & (df.input_type == "latent"), "rmse_test"].iloc[0],
                color="cornflowerblue",
                ls="-",
                lw=2,
            )
            # plot_rmse(
            #     a=a,
            #     data=df.loc[mask_train_size & (df.input_type == "raw"), "rmse_test"].iloc[0],
            #     color="cornflowerblue",
            #     ls=":",
            #     lw=2,
            # )
            if side_legend:
                a.legend(
                    bbox_to_anchor=(1.05, 1,),
                    loc=legend_loc,
                    fontsize=fs_legend,
                    ncols=legend_cols,
                )
            else:
                a.legend(
                    loc=legend_loc,
                    ncols=legend_cols,
                )

        a.plot(
            [SEQ_LEN - 1, SEQ_LEN - 1,],
            [0, 1,],
            "k:",
        )
        if (row == 0) and (col == 0):
            a.set_yticks(
                [0, 0.5, 1,],
                labels=[0, "", 1],
                fontsize=FS_TICKS,
            )
            a.set_xticks(
                [0, 384, 768,],
                labels=[0, 384, 768,],
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
        elif i == n_members[consortium]:
            a.set_yticks(
                [0, 0.5, 1,],
                labels=[0, "", 1],
                fontsize=FS_TICKS,
            )
            a.set_xticks(
                [0, 384, 768,],
                labels=[],
                fontsize=FS_TICKS,
            )
            a.set_ylabel(
                "Accuracy",
                fontsize=fs_accuracy,
            )
            a.set_ylim(0, 1.1)
        else:
            a.set_yticks(
                [0, 0.5, 1,],
                labels=[],
            )
            a.set_xticks(
                [0, 384, 768,],
                labels=[],
            )
        a.set_xticks(
            [128, 256, 512, 640,],
            minor=True,
        )

        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)
    ax.axis("off")


def initialize():
    df = load_data()
    consortia = get_consortia_for_plotting()
    train_size = 12800
    input_type = "latent"
    tgt_type = "latent"
    cross_val = 0
    data = np.load(
        DIR_CACHE_CONSORTIA
        / f"future_outlook_{input_type}_{consortia[1]}_{tgt_type}_{MODEL_TYPE}_{Z_DIM}_{cross_val}_{train_size}_{NOW_TEXT}.npz",
    )
    return df, consortia, data


def main(
    fs_panel,
    fw_panel,
):
    df, consortia, data = initialize()
    fig, ax_dict = get_axes()

    TRAIN_SIZE = 12800

    print(data)
    plot_pipeline(
        a=ax_dict["A"].inset_axes(
            [0.05, 0, 0.4, 0.8,],
        ),
        data=data,
    )

    show_sliding_window(
        a=ax_dict["A"].inset_axes(
            [0.54, 0, 0.4, 0.8,],
        ),
        y=data["w_test_pred"][0, :, :].T,
    )

    for i, ax_label in enumerate(["B", "D"]):
        ax = ax_dict[ax_label].inset_axes(
            [0.05, 0.05, 0.95, 0.95,],
        )
        ax_dict[ax_label].axis("off")
        plot_forecasts(
            ax=ax,
            df=df,
            consortium=consortia[i],
            p_y=0.02,
            p_x=0.01,
            side_legend=True,
            fs_legend=10,
            legend_cols=1,
            legend_loc="upper left",
            fs_accuracy=14,
            train_size=TRAIN_SIZE,
        )

    for x, y, label in [
        (0.02, 0.97, "A",),
        (0.5, 0.97, "B",),
        (0.02, 0.68, "C",),
        (0.02, 0.39, "D",),
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
        / "fig_4.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_4.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_4.svg",
    )
    plt.close()
