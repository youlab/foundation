import matplotlib.pyplot as plt
import numpy as np

from applications.utils.latents import reconstruct
from config import SEQ_LEN
from figs.utils.symbols import (
    get_vae_patches,
    add_arrow,
    add_rectangle,
)
from data.utils import get_data
from ml.utils.load_models import load_default_model


def format_small_axis(
    a,
    include_ticks=False,
):
    for spine in ["top", "right",]:
        a.spines[spine].set_visible(False)
    a.set_ylim(
        0.04,
        1.1,
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


def main(use_fm_output):
    model = load_default_model()
    data_main, idx = get_data(category="all")

    p_x=20
    p_y=0.15
    lw=5
    fs_text=14
    fw_text="bold"

    y = data_main["y"][idx["experimental"]["zach_growth_curves_2023-03-27_3_y.npy"]["y_all_i"]]
    reconstruction = reconstruct(
        model=model,
        x=y.reshape(1, 1, -1),
        batch_size=1,
        is_vae=True,
        is_pca=False,
    )[0, 0, :]

    if use_fm_output:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7,),)

        w1 = 0.21
        w2 = 0.08
        p_x = 0.01
        p_y = 0.1
        p_outer = 0.04
        h_upper = 0.3
        a_fm_input_curve = ax.inset_axes([p_outer, 1 - h_upper-p_y, w1, h_upper,])
        a_fm_input_encoder = ax.inset_axes([p_outer + w1 + p_x, 1 - h_upper-p_y, w2, h_upper,])
        a_fm_output_curve = ax.inset_axes([1-w1-p_outer, (1-h_upper)/2, w1, h_upper,])
        a_fm_output_decoder = ax.inset_axes([1-w1-w2-p_x-p_outer, (1-h_upper)/2, w2, h_upper,])

        a_fm_input_curve.plot(
            np.arange(SEQ_LEN),
            y / y.max(),
            color="cornflowerblue",
            lw=lw,
        )

        a_fm_input_curve.text(
            127,
            0.06,
            "normalized",
            color="cornflowerblue",
            fontweight=fw_text,
            fontsize=fs_text,
            verticalalignment="top",
            horizontalalignment="right",
        )
        format_small_axis(a=a_fm_input_curve)
        a_fm_output_curve.plot(
            np.arange(SEQ_LEN),
            reconstruction / y.max(),
            color="firebrick",
            lw=lw,
        )
        a_fm_output_curve.text(
            127,
            0.06,
            "predicted",
            color="firebrick",
            fontweight=fw_text,
            fontsize=fs_text,
            verticalalignment="top",
            horizontalalignment="right",
        )
        format_small_axis(a=a_fm_output_curve)

        get_vae_patches(
            a=a_fm_input_encoder,
            left=0,
            just_encoder_to_latent=True,
            just_latent_to_decoder=False,
            w_total=1.5,
        )
        get_vae_patches(
            a=a_fm_output_decoder,
            left=0,
            just_encoder_to_latent=False,
            just_latent_to_decoder=True,
            w_total=1.5,
        )

        for a in [
            a_fm_input_encoder,
            a_fm_output_decoder,
        ]:
            a.set_ylim(-0.2, 1.2)
            a.axis("off")

        right = add_rectangle(
            a=ax,
            xy=(0.7, 0.33,),
            width=0.1,
            height=0.5,
            label="task-specific\nmodel",
            fw="semibold",
        )
        ax.text(
            0.75,
            0.3,
            "scikit-learn model",
            fontsize=12,
            fontweight="medium",
            horizontalalignment="center",
            verticalalignment="top",
        )
        

        fs = 16
        ax.text(
            0.3,
            0.8 * 7 / 12,
            "task-specific\ninputs",
            fontsize=fs,
            fontweight="semibold",
            horizontalalignment="center",
            verticalalignment="top",
        )
        ax.text(
            0.3,
            0.6 * 7 / 12,
            "$y_{max}$\nstrain\nmedia\netc.",
            fontsize=fs * 0.9,
            fontweight="medium",
            horizontalalignment="center",
            verticalalignment="top",
        )

        ax.text(
            1.26,
            0.9,
            "task-specific\noutputs",
            fontsize=fs,
            fontweight="semibold",
            horizontalalignment="center",
            verticalalignment="top",
        )

        ax.text(
            0.3,
            2 * 7 / 12,
            "foundation model\ninputs",
            fontsize=fs,
            fontweight="semibold",
            horizontalalignment="center",
            verticalalignment="top",
        )

        y1, y2 = 0.8, 0.7
        y3 = 0.6
        y4, y5 = 0.3,0.5
        x1a = 0.52
        x1b = 0.45
        x2 = 0.65

        width = 0.01
        head_width=0.04
        head_length=0.05

        arrow_color = "dimgray"
        add_arrow(
            a=ax,
            x=x1a,
            y=y1,
            dx=x2-x1a,
            dy=y2-y1,
            color=arrow_color,
            width=width,
            head_width=head_width,
            head_length=head_length,
        )
        add_arrow(
            a=ax,
            x=x1b,
            y=y4,
            dx=x2-x1b,
            dy=y5-y4,
            color=arrow_color,
            width=width,
            head_width=head_width,
            head_length=head_length,
        )

        add_arrow(
            a=ax,
            x=1.5 - x2,
            y=1.1667/2,
            dx=0.1,
            dy=0,
            color=arrow_color,
            width=width,
            head_width=head_width,
            head_length=head_length,
        )

        ax.text(
            1.26,
            0.35,
            "uses foundation model",
            fontsize=fs * 0.8,
            fontweight="medium",
            horizontalalignment="center",
            verticalalignment="top",
        )

        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.1667,)
        ax.axis("off")
        
        return
    fig, ax = plt.subplots(1, 1, figsize=(9, 7,),)

    w1 = 0.21
    w2 = 0.08
    p_x = 0.01
    p_y = 0.1
    p_outer = 0.04
    h_upper = 0.3
    a_fm_input_curve = ax.inset_axes([p_outer, 1 - h_upper-p_y, w1, h_upper,])
    a_fm_input_encoder = ax.inset_axes([p_outer + w1 + p_x, 1 - h_upper-p_y, w2, h_upper,])

    a_fm_input_curve.plot(
        np.arange(SEQ_LEN),
        y / y.max(),
        color="cornflowerblue",
        lw=lw,
    )

    a_fm_input_curve.text(
        127,
        0.06,
        "normalized",
        color="cornflowerblue",
        fontweight=fw_text,
        fontsize=fs_text,
        verticalalignment="top",
        horizontalalignment="right",
    )
    format_small_axis(a=a_fm_input_curve)
    get_vae_patches(
        a=a_fm_input_encoder,
        left=0,
        just_encoder_to_latent=True,
        just_latent_to_decoder=False,
        w_total=1.5,
    )

    for a in [
        a_fm_input_encoder,
    ]:
        a.set_ylim(-0.2, 1.2)
        a.axis("off")


    right = add_rectangle(
        a=ax,
        xy=(0.7, 0.33,),
        width=0.1,
        height=0.5,
        label="task-specific\nmodel",
        fw="semibold",
    )
    ax.text(
        0.75,
        0.3,
        "scikit-learn model",
        fontsize=12,
        fontweight="medium",
        horizontalalignment="center",
        verticalalignment="top",
    )
    

    fs = 16
    ax.text(
        0.3,
        0.8 * 7 / 12,
        "task-specific\ninputs",
        fontsize=fs,
        fontweight="semibold",
        horizontalalignment="center",
        verticalalignment="top",
    )
    ax.text(
        0.3,
        0.6 * 7 / 12,
        "$y_{max}$\nstrain\nmedia\netc.",
        fontsize=fs * 0.9,
        fontweight="medium",
        horizontalalignment="center",
        verticalalignment="top",
    )

    ax.text(
        1.26,
        0.75,
        "task-specific\noutputs",
        fontsize=fs,
        fontweight="semibold",
        horizontalalignment="center",
        verticalalignment="top",
    )
    ax.text(
        1.26,
        0.65,
        "treatment\nresistance\nconcentration\nforecast",
        fontsize=fs * 0.9,
        fontweight="medium",
        horizontalalignment="center",
        verticalalignment="top",
    )

    ax.text(
        0.3,
        2 * 7 / 12,
        "foundation model\ninputs",
        fontsize=fs,
        fontweight="semibold",
        horizontalalignment="center",
        verticalalignment="top",
    )

    y1, y2 = 0.8, 0.7
    y3 = 0.6
    y4, y5 = 0.3,0.5
    x1a = 0.52
    x1b = 0.45
    x2 = 0.65

    width = 0.01
    head_width=0.04
    head_length=0.05

    arrow_color = "dimgray"
    add_arrow(
        a=ax,
        x=x1a,
        y=y1,
        dx=x2-x1a,
        dy=y2-y1,
        color=arrow_color,
        width=width,
        head_width=head_width,
        head_length=head_length,
    )
    add_arrow(
        a=ax,
        x=x1b,
        y=y4,
        dx=x2-x1b,
        dy=y5-y4,
        color=arrow_color,
        width=width,
        head_width=head_width,
        head_length=head_length,
    )

    add_arrow(
        a=ax,
        x=1.5 - x2,
        y=1.1667/2,
        dx=0.2,
        dy=0,
        color=arrow_color,
        width=width,
        head_width=head_width,
        head_length=head_length,
    )

    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.1667,)
    ax.axis("off")
