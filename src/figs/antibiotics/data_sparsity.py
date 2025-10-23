from matplotlib.patches import FancyArrow

from figs.utils.symbols import (
    add_arrow,
    plot_rounded_rectangle,
)
import matplotlib.pyplot as plt

from figs.utils.cross_validation import (
    plot_replicate_row,
    add_train_test_brackets,
    add_kfold_cv_bracket,
    add_rmse_bracket,
)
from config import DIR_FIGS_PRESENTATION


def plot_cross_validation(
    ax,
    left=0,
    w=0.05,
    h=0.1,
    x_space=0.03,
    y_space=0.03,
    y0=0.8,
    rmse_right_adder=0.9,
    w_train=120,
    w_vert=90,
    marker_size=5,
    h_dots=0.05,
    fs_number=12,
    fs_text=14,
    space_dots=0.03,
    length_a=10,
    k=5,
):
    
    bottom = y0 - (h + y_space) * 2 - space_dots
    x_center = (k/2) * w + ((k-1)/2) * x_space
    y = bottom - h_dots - h - space_dots
    
    y8 = y0 + h / 2
    y7 = y8 - h - y_space
    y6 = y7 - h - y_space
    y1 = y + h / 2
    
    right = left + w * k + x_space * (k - 1)
    right_rmse = left + rmse_right_adder
    
    for i in range(5):
        plot_replicate_row(
            a=ax,
            left=left,
            y=y0 - (h + y_space) * i,
            w=w,
            h=h,
            space=x_space,
            i_test=(k - 1) - i,
            fs_number=fs_number,
            k=k,
        )
    
    add_train_test_brackets(
        a=ax,
        left=left,
        y0=y0,
        w=w,
        h=h,
        x_space=x_space,
        y_space=y_space,
        w_train=w_train,
        fs_text=fs_text,
        length_a=length_a,
        k=k,
    )
    
    add_kfold_cv_bracket(
        a=ax,
        left=left,
        top=y0+h,
        bottom=y,
        h=h,
        y_space=y_space,
        w_bracket=w_vert,
        fs_text=fs_text,
        length_a=length_a,
        label=f"{k}-fold CV",
    )
    
    add_rmse_bracket(
        a=ax,
        left=right_rmse,
        top=y0+h,
        bottom=y,
        h=h,
        y_space=y_space,
        w_bracket=w_vert,
        fs_text=fs_text,
        length_a=length_a,
        label=r"$\overline{accuracy}$",
    )
    ax.set_xlim(-0.2, 1.1,)
    ax.set_ylim(0.15, 1.1)
    ax.axis("off")


def main(ax=ax):
    plt.ion()
    ax.axis("off")
    fs_label=13

    X_SPACE = 0.1
    Y_SPACE = 0.08

    a = ax.inset_axes([0, 0, 0.65, 1,])
    b = ax.inset_axes([0.7, 0, 0.3, 1,])
    ax = a

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
    right = add_arrow(
        a=ax,
        x=left,
        y=0.2,
        dx=0.4,
        dy=0,
        color="w",
    )
    left = right + X_SPACE
    adjustment = 1.2
    plot_rounded_rectangle(
        a=ax,
        x=left,
        y=0.4 + adjustment,
        w=1,
        h=1.6 - Y_SPACE - adjustment,
        label="Train\ndata",
        color="tab:blue",
        fs_label=fs_label - 1,
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
    h_box = 0.32
    mult = 1 / 0.6
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
            color="w",
        )
        left = right + X_SPACE

        right = plot_rounded_rectangle(
            a=ax,
            x=left,
            y=y_box,
            w=1,
            h=h_box,
            label="Train\ndata",
            color="tab:blue",
            fs_label=fs_label,
        )
        plot_rounded_rectangle(
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

    ax.add_patch(
        FancyArrow(
            x=2,
            y=2.25,
            dx=5,
            dy=0,
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
        x=4.5,
        y=2.5,
        s="Increasing training data",
        fontsize=fs_label,
        verticalalignment="center",
        horizontalalignment="center",
        fontweight="medium",
    )

    ax.set_xlim(0, 8.5)
    ax.set_ylim(-0.2, 2.5)
    ax.axis("off")

    plot_cross_validation(
        ax=b,
        left=0,
        w=0.15,
        h=0.1,
        x_space=0.03,
        y_space=0.05,
        y0=0.8,
        rmse_right_adder=0.9,
        w_train=50,
        w_vert=70,
        marker_size=3,
        h_dots=0.07,
        space_dots=0.05,
        fs_number=12,
        fs_text=12,
        length_a=10,
    )
