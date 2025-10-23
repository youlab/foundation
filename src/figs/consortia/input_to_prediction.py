import numpy as np

from config import SEQ_LEN
from figs.consortia.main_epsilon import initialize
from figs.utils.colors import get_colors


def plot_input_to_prediction(
    ax,
    x_text1=12,
    y_text=0.8,
    fs_text=18,
    fs_ticks=16,
    fs_label=18,
    horizontal_alignment="left",
    vertical_alignment="top",
    lw=4,
    alpha=0.7,
):
    _, _, data = initialize()
    n_members =  data["w_test"].shape[1]
    colors = get_colors(
        n=n_members,
    )
    x_text2 = x_text1 + SEQ_LEN
    idx = SEQ_LEN * 2
    
    for i_member in range(n_members):
        ax.plot(
            np.arange(SEQ_LEN),
            data["w_test"][0, i_member, idx:idx + SEQ_LEN],
            color=colors[i_member],
            lw=lw,
            alpha=alpha,
        )
        ax.plot(
            np.arange(SEQ_LEN) + SEQ_LEN,
            data["w_test"][0, i_member, idx + SEQ_LEN:idx + SEQ_LEN * 2],
            color=colors[i_member],
            lw=lw,
            alpha=alpha,
        )
    ax.text(
        x=x_text1,
        y=y_text,
        s="Input",
        horizontalalignment=horizontal_alignment,
        verticalalignment=vertical_alignment,
        fontsize=fs_text,
    )
    ax.text(
        x=x_text2,
        y=y_text,
        s="Predicted",
        horizontalalignment=horizontal_alignment,
        verticalalignment=vertical_alignment,
        fontsize=fs_text,
    )
    ax.plot(
        [SEQ_LEN - 1, SEQ_LEN - 1,],
        [0, 1,],
        "k:",
        lw=3,
    )
    ax.set_xlim(0, SEQ_LEN * 2,)
    ax.set_ylim(0, 0.8,)
    for spine in ["top", "right",]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks(
        [0, 128, 256,],
        labels=[0, 128, 256,],
        fontsize=fs_ticks,
    )
    ax.set_yticks(
        [0, 0.4, 0.8,],
        labels=[0, 0.4, 0.8,],
        fontsize=fs_ticks,
    )
    ax.set_xticks(
        [32, 64, 96, 160, 192, 224,],
        minor=True,
    )
    ax.set_yticks(
        [0.1, 0.2, 0.3, 0.5, 0.6, 0.7,],
        minor=True,
    )
    ax.set_ylabel(
        "Cell density",
        fontsize=fs_label,
    )
    ax.set_xlabel(
        "Time",
        fontsize=fs_label,
    )
