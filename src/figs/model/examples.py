from figs.config import FS_TEXT, COLOR1, COLOR2


def panel_examples(
    a,
    y_true,
    y_recon,
    text=None,
    text_x=0,
    text_y=0,
    legend=False,
    fs_legend=10,
):
    a.plot(
        y_true,
        lw=5,
        color=COLOR1,
        alpha=0.5,
        label="Original",
    )
    a.plot(
        y_recon,
        lw=3,
        color=COLOR2,
        alpha=1,
        label="Reconstructed",
    )
    if legend:
        a.legend(
            loc="lower right",
            fontsize=fs_legend,
        )
    if text is not None:
        a.text(
            x=text_x,
            y=text_y,
            s=text,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=FS_TEXT,
            fontweight="medium",
        )
    
    for spine in ["top", "right",]:
        a.spines[spine].set_visible(False)
    a.set_xticks([])
    a.set_yticks([])
    a.set_ylim(0, 1.1)
