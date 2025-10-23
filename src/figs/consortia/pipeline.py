import numpy as np

from figs.utils.colors import get_colors
from figs.utils.symbols import (
    get_vae_patches,
    add_arrow,
    add_regr_rectangle,
)

FS_LABELS = 20
FS_TICKS = 18


def plot_pipeline(
    a,
    data,
    alpha=1,
    y_label=0.7,
    fs_text=16,
):

    n_samples = data["w_test"].shape[1]
    colors = get_colors(n=n_samples)

    SPACE = 0.2
    for i in range(n_samples):
        a.plot(
            np.linspace(0, 1, 128),
            data["w_test"][0, i, :128],
            lw=4,
            c=colors[i],
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

    right = 1
    right = get_vae_patches(
        a=a,
        left=right + SPACE,
        just_encoder_to_latent=True,
    )

    right = add_arrow(
        a=a,
        x=right + SPACE,
        y=0.5,
        dx=0.7,
        dy=0,
    )

    right = add_regr_rectangle(
        a=a,
        xy=(right + SPACE, 0.25,),
        width=0.3,
        height=0.5,
    )

    right = add_arrow(
        a=a,
        x=right + SPACE,
        y=0.5,
        dx=0.7,
        dy=0,
    )

    right = get_vae_patches(
        a=a,
        left=right + SPACE,
        just_latent_to_decoder=True,
    )

    left = right + SPACE
    for i in range(n_samples):
        a.plot(
            np.linspace(left, left + 1, 128),
            data["w_test"][0, i, 128:256],
            lw=4,
            c=colors[i],
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
