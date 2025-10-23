from figs.utils.cross_validation import plot_replicate_box


def plot_boxes(
    a,
    n_cols,
    n_rows,
    left,
    y,
    w,
    h,
    labels,
    x_space,
    y_space,
    fs_number,
    color,
):
    i_label = 0
    for i_col in range(n_cols):
        for i_row in range(n_rows):
            plot_replicate_box(
                a=a,
                x=left + (w + x_space) * i_col,
                y=y + (h + y_space) * i_row,
                w=w,
                h=h,
                number=labels[i_label],
                color=color,
                fs_number=fs_number,
            )
            i_label += 1


def plot_experiment_conditions(
    a,
    fs1=10,
    fs2=11,
    fs_symbol=24,
    fs_text=14,
):
    space = 0.17
    
    plot_boxes(
        a=a,
        n_cols=1,
        n_rows=2,
        left=0.05,
        y=0.27,
        w=0.16,
        h=0.2,
        labels=["water", "soil",],
        x_space=0.03,
        y_space=0.06,
        fs_number=fs1,
        color="tab:green",
    )
    
    plot_boxes(
        a=a,
        n_cols=1,
        n_rows=3,
        left=0.21 + space,
        y=0.08,
        w=0.25,
        h=0.24,
        labels=["oatmeal", "oatmeal-\npeptone", "peptone",],
        x_space=0.03,
        y_space=0.06,
        fs_number=fs1,
        color="lemonchiffon",
    )
    
    plot_boxes(
        a=a,
        n_cols=2,
        n_rows=4,
        left=0.21 + 0.25 + space * 2,
        y=0.19,
        w=0.06,
        h=0.12,
        labels=[7, 5, 3, 1, 8, 6, 4, 2,],
        x_space=0.04,
        y_space=0.06,
        fs_number=fs2,
        color="cornflowerblue",
    )
    a.text(
        x=0.21 + space / 2,
        y=0.5,
        s=r"$\times$",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fs_symbol,
        fontweight="bold",
    )
    
    a.text(
        x=0.21 + space + 0.25 + space / 2,
        y=0.5,
        s=r"$\times$",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fs_symbol,
        fontweight="bold",
    )
    a.text(
        1.,
        0.5,
        r"$=$",
        fontsize=fs_symbol,
        horizontalalignment="left",
        verticalalignment="center",
    )

    a.text(
        1.23,
        0.5,
        "48\n exp.",
        fontsize=fs_text,
        horizontalalignment="center",
        verticalalignment="center",
    )
    a.axis("off")
