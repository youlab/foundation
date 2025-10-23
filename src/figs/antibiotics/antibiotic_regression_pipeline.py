import numpy as np

from figs.utils.sample_curve import SAMPLE_CURVE
from figs.utils.symbols import (
    add_arrow,
    add_regr_rectangle,
    get_vae_patches,
    plot_rounded_rectangle,
)


def plot_antibiotic_regression_pipeline(
    ax,
    h_box=0.3,
    w_box=0.8,
    p_right=0.2,
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
    add_regr_rectangle(
        a=ax,
        xy=(left, 0.75 - h_gold / 2),
        width=0.3,
        height=h_gold,
    )
    right = add_regr_rectangle(
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
    
    for i in range(2):
        if i == 0:
            bottom =0.6
        else:
            bottom = 0.1
        
        a = ax.inset_axes(
            [left / (left + w_box + p_right), bottom, w_box / (left + w_box + p_right), h_box,],
        )
        a.scatter(
            [0.1, 0.4, 0.7,],
            [0.1, 0.4, 0.7,],
            s=100,
        )
        a.set_xlabel("True\nconcentration")
        a.set_ylabel("Predicted\nconcentration")
        a.set_xticks([])
        a.set_yticks([])
        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)
        a.set_xlim(0, 0.8)
        a.set_ylim(0, 0.8)
    
    ax.set_xlim(0, 4.9)
    ax.set_ylim(0.4, 2.1)
    ax.axis("off")
