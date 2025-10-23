import numpy as np

from figs.utils.colors import get_colors
from figs.utils.data_mapping import (
    get_mapping_and_samples_for_experimental,
    get_mapping_and_samples_for_simulation,
)


def plot_curves(
    ax,
    data,
    y,
    n_samples,
    color,
    alpha,
):
    if len(data) > n_samples:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(
            data,
            n_samples,
        )
        y = y[indices].T
    else:
        y = y[data].T

    for j in range(y.shape[1]):
        ax.plot(
            y[:, j],
            alpha=alpha[j],
            color=color,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_experiment(
    ax,
    data,
    samples,
    mapping,
    i,
    n_samples,
    consortia_names,
    colors,
):
    name = mapping[i]
    plot_curves(
        ax=ax,
        data=samples[name],
        y=data["y"],
        n_samples=n_samples,
        color=colors[1] if name in consortia_names else colors[0],
        alpha=np.logspace(-0.7, -0.2, n_samples),
    )


def plot_simulation(
    ax,
    data,
    samples,
    mapping,
    i,
    n_samples,
    consortia_names,
    colors,
):
    i -= 28
    name = mapping[i]
    plot_curves(
        ax=ax,
        data=samples[name],
        y=data["y"],
        n_samples=n_samples,
        color=colors[3] if name in consortia_names else colors[2],
        alpha=np.logspace(-0.7, -0.2, n_samples),
    )


def plot_data_sources(
    ax,
    p_y=0.02,
    p_x=0.01,
    n_samples=25,
    n_cols=8,
    colors=get_colors(n=8),
    show_numbers=True,
):
    consortia_names = {
        "feilunconsortia",
        "fujita",
        "huge",
        "schluter",
        "chaotic",
    }

    n_panels = 31
    n_rows = n_panels // n_cols + 1
    x_step = 1 / n_cols
    y_step = 1 / n_rows

    data, samples, mapping = get_mapping_and_samples_for_experimental(consortia_names=consortia_names)
    plot_fcn = plot_experiment

    i_ax = 0
    for i_row in reversed(range(n_rows)):
        for i_col in range(n_cols):
            if i_ax == 31:
                break
            if i_ax == 28:
                data, samples, mapping = get_mapping_and_samples_for_simulation(consortia_names=consortia_names)
                plot_fcn = plot_simulation
    
            a = ax.inset_axes(
                [i_col * x_step + p_x, i_row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
            )
            plot_fcn(
                ax=a,
                data=data,
                samples=samples,
                mapping=mapping,
                i=i_ax,
                n_samples=n_samples,
                consortia_names=consortia_names,
                colors=colors,
            )
            if show_numbers:
                a.text(130, 0.9, i_ax +1, color="k")
                a.set_xlim(0, 140,)
            else:
                a.set_xlim(0, 127,)
            ax.set_ylim(0, 1.01)

            i_ax += 1
    ax.axis("off")
