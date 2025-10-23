import numpy as np

from applications.utils.latents import reconstruct
from config import SEQ_LEN
from figs.utils.symbols import (
    add_arrow,
    get_vae_patches,
)


def plot_reconstruction_pipeline(
    ax,
    data,
    idx,
    model,
    fs_labels=16,
    p_x=20,
    p_y=0.15,
    lw=5,
    ax_bounds_1=[0.05, 0.1, 0.4, 0.8,],
    ax_bounds_2=[0.55, 0.1, 0.4, 0.8,],
    fs_text=14,
    fw_text="bold",
    fw_labels="medium",
    fs_vae_label=12,
    fw_vae_label="medium",
    ylim_min=-0.1,
    ylim_max=1.0,
    include_ticks=False,
    label_shapes=True,
    label_sizes=False,
):
    y = data["y"][idx["experimental"]["zach_growth_curves_2023-03-27_3_y.npy"]["y_all_i"]]
    reconstruction = reconstruct(
        model=model,
        x=y.reshape(1, 1, -1),
        batch_size=1,
        is_vae=True,
        is_pca=False,
    )[0, 0, :]

    def format_small_axis(a):
        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)
        a.set_ylim(
            0.04,
            1.2,
        )
        if include_ticks:
            a.set_xticks(
                [0, 64, 128,],
                labels=[],
            )
            a.set_xticks(
                [32, 96,],
                minor=True,
            )
            a.set_yticks(
                [0, 0.5, 1,],
                labels=[],
            )
            a.set_yticks(
                [0.25, 0.75,],
                minor=True
            )
        else:
            a.set_yscale("log")
            a.set_xticks([])
            a.set_yticks([])
            a.set_yticks([], minor=True,)
        
    a = ax.inset_axes(
        ax_bounds_1,
    )
    a.plot(
        np.arange(SEQ_LEN),
        y / y.max(),
        color="cornflowerblue",
        lw=lw,
    )
    a.text(
        -5,
        1.2,
        "original",
        color="cornflowerblue",
        fontweight=fw_text,
        fontsize=fs_text,
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    a.set_xlabel(
        "Time",
        fontsize=fs_labels,
        fontweight=fw_labels,
    )
    a.set_ylabel(
        "Cell density",
        fontsize=fs_labels,
        fontweight=fw_labels,
    )
    format_small_axis(a=a)
    a = ax.inset_axes(
        ax_bounds_2,
    )
    a.plot(
        np.arange(SEQ_LEN),
        reconstruction / y.max(),
        color="firebrick",
        lw=lw,
    )
    a.text(
        -5,
        1.2,
        "reconstruction",
        color="firebrick",
        fontweight=fw_text,
        fontsize=fs_text,
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    format_small_axis(a=a)
    get_vae_patches(
        a=ax,
        left=SEQ_LEN * 1.4 + 10 + p_x,
        bottom=p_y,
        w_total=SEQ_LEN * 1.3 - p_x * 2,
        h_total=1 - p_y * 2,
        label_shapes=label_shapes,
        fs_text=fs_vae_label,
        fw_text=fw_vae_label,
        h_trap2_mult=0.6,
        h_latent_mult=0.4,
    )
    if label_sizes:
        fs_sizes = 20
        fw_sizes = "semibold"
        sizes_left = SEQ_LEN * 1.4 + 10 + p_x
        sizes_w = SEQ_LEN * 1.3 - p_x * 2
        sizes_bottom = 1 - p_y / 2
        sizes_p = 20
        width = 0.02
        head_width = 0.07
        head_length = None
        ax.text(
            x=sizes_left,
            y=sizes_bottom,
            s=128,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=fs_sizes,
            fontweight=fw_sizes,
        )
        ax.text(
            x=sizes_left + sizes_w / 2,
            y=sizes_bottom,
            s=8,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=fs_sizes,
            fontweight=fw_sizes,
        )
        ax.text(
            x=sizes_left + sizes_w,
            y=sizes_bottom,
            s=128,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=fs_sizes,
            fontweight=fw_sizes,
        )
        add_arrow(
            a=ax,
            x=sizes_left + sizes_p,
            y=sizes_bottom,
            dx=sizes_w / 2 - sizes_p * 1.5,
            dy=0,
            width=width,
            head_width=head_width,
            head_length=head_length,
        )
        add_arrow(
            a=ax,
            x=sizes_left + sizes_w / 2 + sizes_p * 0.5,
            y=sizes_bottom,
            dx=sizes_w / 2 - sizes_p * 1.5,
            dy=0,
            width=width,
            head_width=head_width,
            head_length=head_length,
        )
        ax.text(
            x=sizes_left + sizes_w / 2,
            y=0.1,
            s="50.4M trainable\nparameters",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            fontweight=fw_sizes,
        )
    ax.set_xlim(0, SEQ_LEN * 4 - 1)
    ax.set_ylim(ylim_min, ylim_max,)
    ax.axis("off")
