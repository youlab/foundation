from figs.utils.colors import get_colors


def plot_example(
    a,
    data,
    add_ylabel,
    lw_true=8,
    lw_pred=4,
    colors=None,
):
    y_true, y_pred = data
    n_members = y_true.shape[0]
    if colors is None:
        colors = get_colors(n=n_members)
    for member in range(n_members):
        add_legend_label = (member == 0) and add_ylabel
            
        a.plot(
            y_true[member],
            lw=lw_true,
            alpha=0.4,
            color=colors[member],
            label="ground truth" if add_legend_label else "_nolegend",
        )
        
        a.plot(
            y_pred[member],
            lw=lw_pred,
            color=colors[member],
            label="predicted" if add_legend_label else "_nolegend",
        )
