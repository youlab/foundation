import numpy as np

from figs.utils.bacteria import plot_bacteria
from figs.utils.sample_curve import SAMPLE_CURVE
from figs.utils.symbols import (
    add_arrow,
    add_clf_rectangle,
    get_vae_patches,
    plot_rounded_rectangle,
)


def plot_resistance_classification_pipeline(
    ax,
    x_bact=0.75,
    w_bact=0.3,
    h_bact=0.1,
    y_bact_hi=0.67,
    y_bact_lo=0.33,
    fs_label=14,
):
    X_SPACE = 0.1
    Y_SPACE = 0.08
    
    right = plot_rounded_rectangle(
        a=ax,
        x=0.05,
        y=0.4 + Y_SPACE / 2,
        w=1,
        h=1.6-Y_SPACE,
        label="Train\ndata",
        color="tab:blue",
        fs_label=fs_label,
    )
    left = right + X_SPACE

    add_arrow(
        a=ax,
        x=left,
        y=1.65,
        dx=0.4,
        dy=0,
    )
    right = add_arrow(
        a=ax,
        x=left,
        y=0.75,
        dx=0.4,
        dy=0,
    )
    left = right + X_SPACE
    h_vae = 0.6
    w_total = h_vae * 1.5
    get_vae_patches(
        a=ax,
        left=left,
        just_encoder_to_latent=True,
        bottom=0.75 - h_vae / 2,
        w_total=w_total,
        h_total=h_vae,
    )
    
    adjusted_curve = (SAMPLE_CURVE - SAMPLE_CURVE.min()) / (SAMPLE_CURVE.max() - SAMPLE_CURVE.min())
    adjusted_curve = adjusted_curve * h_vae + 1.65 - h_vae / 2
    ax.plot(
        np.linspace(0, w_total * 0.9/1.5, len(SAMPLE_CURVE),) + left,
        adjusted_curve,
        lw=4,
    )
    
    left += w_total * 0.9/1.5 + X_SPACE
    add_arrow(
        a=ax,
        x=left,
        y=1.65,
        dx=0.4,
        dy=0,
    )
    right = add_arrow(
        a=ax,
        x=left,
        y=0.75,
        dx=0.4,
        dy=0,
    )
    left = right + X_SPACE
    
    h_gold = h_vae * 0.7
    add_clf_rectangle(
        a=ax,
        xy=(left, 0.75 - h_gold / 2),
        width=0.3,
        height=h_gold,
    )
    right = add_clf_rectangle(
        a=ax,
        xy=(left, 1.65 - h_gold / 2),
        width=0.3,
        height=h_gold,
    )
    left = right + X_SPACE
    
    add_arrow(
        a=ax,
        x=left,
        y=1.65,
        dx=0.4,
        dy=0,
    )
    right = add_arrow(
        a=ax,
        x=left,
        y=0.75,
        dx=0.4,
        dy=0,
    )
    left = right + X_SPACE
    
    plot_bacteria(
        ax=ax.inset_axes([x_bact, y_bact_hi, w_bact, h_bact,]),
    )
    
    plot_bacteria(
        ax=ax.inset_axes([x_bact, y_bact_lo, w_bact, h_bact,]),
    )

    ax.set_xlim(0, 4.9)
    ax.set_ylim(0.4, 2.1)
    ax.axis("off")
