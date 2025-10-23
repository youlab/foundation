import numpy as np

from matplotlib.patches import FancyArrowPatch
from figs.utils.colors import get_colors
from figs.utils.symbols import (
    get_vae_patches,
    add_arrow,
    add_clf_rectangle,
)


def add_bracket(
    a,
    left,
    top,
    bottom,
    space,
    w_bracket,
    left_to_right,
    fs_text,
    label,
    length_a=15,
):
    x0_bracket = left + space
    x1_bracket = left + left_to_right
    y_bracket = (top - bottom) / 2 + bottom
    bracketstyle = f"]-, widthA={w_bracket}, lengthA={length_a}"
    a.add_patch(
        FancyArrowPatch(
            (
                x0_bracket,
                y_bracket,
            ),
            (
                x1_bracket,
                y_bracket,
            ),
            arrowstyle=bracketstyle,
            color="black",
        )
    )

    x_text = x1_bracket + space 
    a.text(
        x_text,
        y_bracket,
        label,
        fontsize=fs_text,
        horizontalalignment="left",
        verticalalignment="center",
    )


def plot_classification_pipeline(
    a,
    data,
    w,
    fs_text=16,
):

    n_samples = data["w_test"].shape[1]
    colors = get_colors(n=n_samples)

    SPACE = 0.2
    for i in range(n_samples):
        a.plot(
            np.linspace(0, 1, 128),
            data["w_test"][0, i, :128],
            lw=3,
            c=colors[i],
            alpha=0.6,
        )
    a.text(
        x=0.5,
        y=0.9,
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
        dx=0.4,
        dy=0,
    )

    right = add_clf_rectangle(
        a=a,
        xy=(right + SPACE, 0.25,),
        width=0.3,
        height=0.5,
    )

    right = add_arrow(
        a=a,
        x=right + SPACE,
        y=0.5,
        dx=0.4,
        dy=0,
    )

    left = right + SPACE
    
    colors = get_colors(n=n_samples,)
    for member in range(n_samples):
        a.scatter(
            left + 0.03 * member,
            data["w_test"][0, member, -1],
            color=colors[member],
            s=100,
        )
    a.plot(
        [left, left + 0.03 * n_samples,],
        [0.1, 0.1,],
        "k:",
    )

    left = left + 0.03 * n_samples + SPACE
    add_bracket(
        a=a,
        left=left,
        top=1,
        bottom=0.1,
        space=0.02,
        left_to_right=0.5,
        w_bracket=w,
        fs_text=fs_text,
        label=r"$t_{final} \geq 0.1$",
        length_a=10,
    )

    add_bracket(
        a=a,
        left=left,
        top=0.1,
        bottom=0,
        space=0.02,
        left_to_right=0.2,
        w_bracket=w/9,
        fs_text=fs_text,
        label=r"$t_{final} < 0.1$",
        length_a=10,
    )
    a.axis("off")
