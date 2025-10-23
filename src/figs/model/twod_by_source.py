import joblib

import numpy as np

from config import DIR_DATA
from figs.utils.colors import get_colors
from figs.utils.data_mapping import (
    get_mapping_and_samples_for_experimental,
    get_mapping_and_samples_for_simulation,
)
from figs.model.twod_visualization import main as run_tsne


def plot_twod_by_source(
    ax,
    model,
    fs_text= 16,
    fw_text="semibold",
    x_text=-140,
    y_text=140,
    vertical_alignment="center",
    horizontal_alignment="center",
    p_x=0.05,
    p_y=0.05,
    left=0,
):
    run_tsne(
        model=model,
    )
    ax.axis("off")
    w = (1 - p_x * 4) / 2
    h = (1 - p_y * 4) / 2
    a = []
    for i_row in range(2):
        for i_col in range(2):
            a.append(
                ax.inset_axes(
                    [
                        p_x + (w + p_x * 2) * i_col + left,
                        p_y + (h + p_y * 2) * i_row,
                        w,
                        h,
                    ]
                )
            )
    ax = np.array(a)
    with open(DIR_DATA / "tsne_model.joblib", "rb") as fp:
        tsne_model = joblib.load(fp)
    latents_embedded = tsne_model.embedding_
    colors = get_colors(n=8)

    consortia_names = {
        "feilunconsortia",
        "fujita",
        "huge",
        "schluter",
        "chaotic",
    }

    _, samples, _ = get_mapping_and_samples_for_experimental(consortia_names=consortia_names)

    labels = [
        "Clonal\nexperiment",
        "Consortia\nexperiment",
        "Clonal\nsimulation",
        "Consortia\nsimulation",
    ]
    for i, a in enumerate(ax):
        a.text(
            x=x_text,
            y=y_text,
            s=labels[i],
            verticalalignment=vertical_alignment,
            horizontalalignment=horizontal_alignment,
            fontsize=fs_text,
            fontweight=fw_text,
        )
    
        a.scatter(
            latents_embedded[:, 0],
            latents_embedded[:, 1],
            alpha=0.01,
            color="lavender",
            s=1,
        )
        a.axis("off")

    for name in samples.keys():
        if name in consortia_names:
            color = colors[1]
            a = ax[1]
        else:
            color = colors[0]
            a = ax[0]
        a.scatter(
            latents_embedded[samples[name], 0],
            latents_embedded[samples[name], 1],
            alpha=0.1,
            color=color,
            s=1,
        )

    _, samples, _ = get_mapping_and_samples_for_simulation(consortia_names=consortia_names)

    for name in samples.keys():
        if name in consortia_names:
            color = colors[3]
            a = ax[3]
        else:
            color = colors[2]
            a = ax[2]
        a.scatter(
            latents_embedded[samples[name], 0],
            latents_embedded[samples[name], 1],
            alpha=0.1,
            color=color,
            s=1,
        )
