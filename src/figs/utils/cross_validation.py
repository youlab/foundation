from matplotlib.patches import FancyArrowPatch, BoxStyle, FancyBboxPatch


def plot_replicate_box(
    a,
    x,
    y,
    w,
    h,
    number,
    color,
    fs_number,
):

    a.text(
        x + w/2,
        y + h/2,
        number,
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fs_number,
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


def plot_replicate_row(
    a,
    left,
    y,
    w,
    h,
    space,
    i_test,
    fs_number,
    k=8,
):
    for i in range(k):
        if i < i_test:
            x = left + (w + space) * i
        elif i == i_test:
            x = left + (w + space) * (k - 1)
        else:
            x = left + (w + space) * (i - 1)

        plot_replicate_box(
            a=a,
            x=x,
            y=y,
            w=w,
            h=h,
            number=i + 1,
            color="tab:orange" if i == i_test else "tab:blue",
            fs_number=fs_number,
        )


def add_train_test_brackets(
    a,
    left,
    y0,
    w,
    h,
    x_space,
    y_space,
    fs_text,
    w_train=140,
    length_a=15,
    k=8,
):
    h0_bracket = y0 + h + y_space * 3
    h1_bracket = y0 + h * 2 + y_space * 2
    x_bracket_train = left + ((k-1) * w + (k-2) * x_space) / 2
    x_bracket_test = left + (k-1) * w + (k-1) * x_space + w / 2
    bracketstyle_train = f"]-, widthA={w_train}, lengthA={length_a}"
    bracketstyle_test = f"]-, widthA={w_train / ((k-1) * w + (k-2) * x_space) * w}, lengthA={length_a}"
    a.add_patch(
        FancyArrowPatch(
            (
                x_bracket_train,
                h0_bracket,
            ),
            (
                x_bracket_train,
                h1_bracket,
            ),
            arrowstyle=bracketstyle_train, 
            color="black",
        )
    )   
    a.add_patch(
        FancyArrowPatch(
            (
                x_bracket_test,
                h0_bracket,
            ),
            (
                x_bracket_test,
                h1_bracket,
            ),
            arrowstyle=bracketstyle_test, 
            color="black",
        )
    )

    h_text = h1_bracket + y_space 
    for x, text in [(x_bracket_train, "train",), (x_bracket_test, "test",),]:
        a.text(
            x,
            h_text,
            text,
            fontsize=fs_text,
            horizontalalignment="center",
            verticalalignment="center",
        )


def add_kfold_cv_bracket(
    a,
    left,
    top,
    bottom,
    h,
    y_space,
    w_bracket,
    fs_text,
    length_a=15,
    label="8-fold CV",
):
    x0_bracket = left - (y_space * 3)
    x1_bracket = left - (h + y_space * 2)
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

    x_text = x1_bracket - y_space 
    a.text(
        x_text,
        y_bracket,
        label,
        rotation="vertical",
        fontsize=fs_text,
        horizontalalignment="center",
        verticalalignment="center",
    )


def add_rmse_bracket(
    a,
    left,
    top,
    bottom,
    h,
    y_space,
    w_bracket,
    fs_text,
    length_a=15,
    label=r"$\overline{RMSE}$",
):
    x0_bracket = left + (y_space * 3)
    x1_bracket = left + (h + y_space * 2)
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

    x_text = x1_bracket + y_space 
    a.text(
        x_text,
        y_bracket,
        label,
        fontsize=fs_text,
        horizontalalignment="left",
        verticalalignment="center",
    )


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
):
    
    bottom = y0 - (h + y_space) * 2 - space_dots
    x_center = 4 * w + 3.5 * x_space
    y = bottom - h_dots - h - space_dots
    
    y8 = y0 + h / 2
    y7 = y8 - h - y_space
    y6 = y7 - h - y_space
    y1 = y + h / 2
    
    right = left + w * 8 + x_space * 7
    right_rmse = left + rmse_right_adder
    
    for i in range(3):
        plot_replicate_row(
            a=ax,
            left=left,
            y=y0 - (h + y_space) * i,
            w=w,
            h=h,
            space=x_space,
            i_test=7 - i,
            fs_number=fs_number,
        )
    
    ax.scatter(
        [x_center] * 3,
        [bottom, bottom - h_dots / 2, bottom - h_dots,],
        marker="o",
        color="k",
        s=marker_size,
    )

    plot_replicate_row(
        a=ax,
        left=left,
        y=y,
        w=w,
        h=h,
        space=x_space,
        i_test=0,
        fs_number=fs_number,
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
    )
    
    for i, y_rmse in [(8, y8), (7, y7), (6, y6), (1, y1),]:
        if i == 8:
            text = r"$RMSE_8$"
        elif i == 7:
            text = r"$RMSE_7$"
        elif i == 6:
            text = r"$RMSE_6$"
        elif i == 1:
            text = r"$RMSE_1$"
        ax.annotate(
            text=text,
            xy=(
                right + x_space,
                y_rmse,
            ),
            xytext=(
                right + x_space * 4,
                y_rmse,
            ),
            fontsize=fs_text,
            arrowprops=dict(arrowstyle="<|-", color="k",),
            verticalalignment="center",
        )
