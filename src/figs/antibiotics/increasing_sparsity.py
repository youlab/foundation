from matplotlib.patches import FancyArrow

from figs.utils.symbols import (
    add_arrow,
    plot_rounded_rectangle,
)


def plot_increasing_sparsity(
    ax,
    text_angle,
    fs_label=14,
):
    X_SPACE = 0.1
    Y_SPACE = 0.08
    
    right = plot_rounded_rectangle(
        a=ax,
        x=X_SPACE / 2,
        y=Y_SPACE / 2,
        w=1,
        h=2 - Y_SPACE,
        label="All\ndata",
        color="gray",
        fs_label=fs_label,
    )
    left = right + X_SPACE
    add_arrow(
        a=ax,
        x=left,
        y=1.2,
        dx=0.4,
        dy=0,
    )
    right = add_arrow(
        a=ax,
        x=left,
        y=0.2,
        dx=0.4,
        dy=0,
    )
    left = right + X_SPACE
    plot_rounded_rectangle(
        a=ax,
        x=left,
        y=0.4 + Y_SPACE / 2,
        w=1,
        h=1.6-Y_SPACE,
        label="Train\ndata",
        color="tab:blue",
        fs_label=fs_label,
    )
    right = plot_rounded_rectangle(
        a=ax,
        x=left,
        y=Y_SPACE / 2,
        w=1,
        h=0.4-Y_SPACE,
        label="Test",
        color="tab:orange",
        fs_label=fs_label,
    )
    left = right + X_SPACE
    
    y_arrow = 1.2
    h_box = 1.6
    mult = 0.6
    y_box = 2 - h_box * mult
    for i in range(3):
        
        h_box *= mult
        y_box = 2 - h_box - Y_SPACE
        right = add_arrow(
            a=ax,
            x=left,
            y=(2 - y_box) / 2 + y_box - Y_SPACE / 2,
            dx=0.4,
            dy=0,
        )
        left = right + X_SPACE
        
        right = plot_rounded_rectangle(
            a=ax,
            x=left,
            y=y_box,
            w=1,
            h=h_box,
            label="Train\ndata" if i < 2 else "Train",
            color="tab:blue",
            fs_label=fs_label,
        )
    
        left = right + X_SPACE

    ax.add_patch(
        FancyArrow(
            x=3.7,
            y=0.6,
            dx=3.3,
            dy=0.5,
            width=0.07,
            length_includes_head=True,
            head_width=0.2,
            head_length=0.2,
            shape='full',
            overhang=0,
            head_starts_at_zero=False,
            color="k",
        )
    )
    ax.text(
        x=5.5,
        y=0.57,
        s="Increasing sparsity",
        fontsize=fs_label,
        verticalalignment="center",
        horizontalalignment="center",
        fontweight="semibold",
        rotation=text_angle,
    )
    
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 2)
    ax.axis("off")
