import numpy as np
from matplotlib.patches import Polygon, Rectangle, FancyArrow, FancyBboxPatch, BoxStyle


def get_vae_patches(
    a,
    left,
    just_encoder_to_latent=False,
    just_latent_to_decoder=False,
    bottom=0,
    w_total=1.5,
    h_total=1,
    h_trap2_mult=0.8,
    h_latent_mult=0.6,
    label_shapes=False,
    fs_text=12,
    fw_text="medium",
):
    colors = {
        "encoder": "cornflowerblue",
        "latent": "rebeccapurple",
        "decoder": "firebrick",
    }

    H_TRAP1 = 1 * h_total
    H_TRAP2 = h_trap2_mult * h_total
    H_LATENT = h_latent_mult * h_total

    W_TRAP = 0.5 * w_total / 1.5
    W_LATENT = 0.3 * w_total / 1.5

    SPACE = 0.1 * w_total / 1.5

    bottom_latent = (H_TRAP1 - H_LATENT) / 2 + bottom
    bottom_trap2 = (H_TRAP1 - H_TRAP2) / 2 + bottom

    if just_encoder_to_latent and just_latent_to_decoder:
        raise ValueError("You may not select both of these")
    elif just_latent_to_decoder:
        left_latent = left
    else:
        left_encoder = left
        left_latent = left_encoder + W_TRAP + SPACE
    
    right_latent = left_latent + W_LATENT

    if just_encoder_to_latent:
        right = right_latent
    else:
        left_decoder = right_latent + SPACE
        right = left_decoder + W_TRAP

    latent_xy = np.array(
        [
            [left_latent, bottom_latent,],
            [left_latent, bottom_latent + H_LATENT,],
            [left_latent + W_LATENT, bottom_latent + H_LATENT,],
            [left_latent + W_LATENT, bottom_latent,],
        ],
    )

    if not just_encoder_to_latent:
        decoder_xy = np.array(
            [
                [left_decoder, bottom_trap2,],
                [left_decoder, bottom_trap2 + H_TRAP2,],
                [left_decoder + W_TRAP, H_TRAP1 + bottom,],
                [left_decoder + W_TRAP, bottom,],
            ],
        )

    if not just_latent_to_decoder:
        encoder_xy = np.array(
            [
                [left_encoder, bottom,],
                [left_encoder, H_TRAP1 + bottom,],
                [left_encoder + W_TRAP, bottom_trap2 + H_TRAP2,],
                [left_encoder + W_TRAP, bottom_trap2,],
            ],
        )

    if just_encoder_to_latent:
        patches = [
            (encoder_xy, colors["encoder"],),
            (latent_xy, colors["latent"],),
        ]
        if label_shapes:
            labels = [
                (encoder_xy, "encoder",),
                (latent_xy, "latent",),
            ]
    elif just_latent_to_decoder:
        patches = [
            (latent_xy, colors["latent"],),
            (decoder_xy, colors["decoder"],),
        ]
        if label_shapes:
            labels = [
                (latent_xy, "latent",),
                (decoder_xy, "decoder",),
            ]
    else:
        patches = [
            (encoder_xy, colors["encoder"],),
            (latent_xy, colors["latent"],),
            (decoder_xy, colors["decoder"],),
        ]
        if label_shapes:
            labels = [
                (encoder_xy, "encoder",),
                (latent_xy, "latent",),
                (decoder_xy, "decoder",),
            ]

    for xy, color in patches:
        a.add_patch(
            Polygon(
                xy=xy,
                closed=True,
                edgecolor="k",
                fill=True,
                facecolor=color,
                alpha=0.6,
                lw=2,
            )
        )
    if label_shapes:
        for xy, label in labels:
            x = xy[0][0] + (xy[2][0] - xy[0][0]) / 2
            y = xy[0][1] + (xy[1][1] - xy[0][1]) / 2
            a.text(
                x=x,
                y=y,
                s=label,
                rotation="vertical",
                fontweight=fw_text,
                fontsize=fs_text,
                verticalalignment="center",
                horizontalalignment="center",
            )
    return right


def add_arrow(
    a,
    x,
    y,
    dx,
    dy,
    width=None,
    head_width=None,
    head_length=None,
    color="k",
):
    a.add_patch(
        FancyArrow(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            width=dx / 7 if (width is None) else width,
            length_includes_head=True,
            head_width=dx * 3 / 7 if (head_width is None) else head_width,
            head_length=dx * 3 / 7 if (head_length is None) else head_length,
            shape='full',
            overhang=0,
            head_starts_at_zero=False,
            color=color,
        )
    )
    return x + dx


def add_regr_rectangle(
    a,
    xy,
    width,
    height,
):
    a.add_patch(
        Rectangle(
            xy=xy,
            width=width,
            height=height,
            facecolor="gold",
            edgecolor="k",
            lw=3,
            alpha=0.6,
        )
    )
    a.text(
        x=xy[0] + width / 2,
        y=xy[1] + height * 1.1,
        s="Regression",
        verticalalignment="bottom",
        horizontalalignment="center",
        fontsize=16,
    )

    return xy[0] + width


def add_clf_rectangle(
    a,
    xy,
    width,
    height,
):
    a.add_patch(
        Rectangle(
            xy=xy,
            width=width,
            height=height,
            facecolor="gold",
            edgecolor="k",
            lw=3,
            alpha=0.6,
        )
    )
    a.text(
        x=xy[0] + width / 2,
        y=xy[1] + height * 1.1,
        s="Classification",
        verticalalignment="bottom",
        horizontalalignment="center",
        fontsize=16,
    )

    return xy[0] + width


def plot_rounded_rectangle(
    a,
    x,
    y,
    w,
    h,
    label,
    color,
    fs_label,
):

    a.text(
        x + w/2,
        y + h/2,
        label,
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fs_label,
        fontweight="medium",
    )
    a.add_patch(
        FancyBboxPatch(
            (x, y), 
            w,
            h,
            edgecolor="k",
            facecolor=color,
            alpha=0.5,
            boxstyle=BoxStyle(
                "Round",
                pad=0.01,
            ),
        ),
    )
    return x + w


def add_rectangle(
    a,
    xy,
    width,
    height,
    label,
    fw,
):
    a.add_patch(
        Rectangle(
            xy=xy,
            width=width,
            height=height,
            facecolor="gold",
            edgecolor="k",
            lw=3,
            alpha=0.6,
        )
    )
    a.text(
        x=xy[0] + width / 2,
        y=xy[1] + height * 1.1,
        s=label,
        verticalalignment="bottom",
        horizontalalignment="center",
        fontsize=16,
        fontweight=fw,
    )

    return xy[0] + width
