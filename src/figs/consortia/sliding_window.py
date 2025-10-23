from matplotlib.patches import FancyArrowPatch

from config import SEQ_LEN
from figs.utils.colors import get_colors


def show_sliding_window(
    a,
    y,
    alpha=0.8,
    lw=4,
):
    colors = get_colors(n=y.shape[1])
    for i in range(y.shape[1]):
        a.plot(
            y[:, i],
            lw=lw,
            color=colors[i],
            alpha=alpha,
        )
    for i in range(1, 6,):
        a.plot(
            [i * SEQ_LEN, i * SEQ_LEN],
            [0, 1,],
            "k:",
        )
        start = (90 + SEQ_LEN * (i - 1), 0.7)
        end = (170 + SEQ_LEN * (i - 1), 0.7)
        
        # Create a curved arrow
        arrow = FancyArrowPatch(
            start, end, 
            connectionstyle="arc3,rad=-0.3",  # Controls curvature
            # arrowstyle="->", 
            mutation_scale=15,  # Size of arrowhead
            color="black",
        )
        a.add_patch(arrow)
    
    a.text(
        x=50,
        y=0.7,
        s="Input",
        fontsize=16,
        horizontalalignment="center",
        verticalalignment="center",
    )

    for spine in ["top", "right",]:
        a.spines[spine].set_visible(False)
    
    a.set_yticks(
        [0, 0.5, 1,],
        labels=[0, "", 1,],
        fontsize=14,
    )

    a.set_xticks(
        [0, 384, 768,],
        labels=[0, 384, 768,],
        fontsize=14,
    )

    a.set_xlim(0, 768,)
    a.set_ylim(0, 1,)
