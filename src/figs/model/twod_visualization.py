import joblib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from config import (
    DIR_DATA,
    DIR_SRC,
    SEQ_LEN,
)
from figs.config import COLOR1
from figs.model.examples import panel_examples
from data.utils import get_data
from applications.utils.latents import (
    get_latents,
    reconstruct,
)


def plot_simple_rectangle(
    main_ax,
    latents_embedded,
    y,
    n_points=6,
    marker_size=100,
    radius_max=100,
):
    if main_ax is None:
        plt.ion()
        fig, ax0 = plt.subplots(figsize=(6, 6,))
        ax_existed = False
        for spine in ["top", "left", "bottom", "right",]:
            ax0.spines[spine].set_visible(False)
        ax0.set_xticks([])
        ax0.set_yticks([])

        main_size = 0.5
        main_plot_height = main_size
        main_plot_bottom = (1 - main_size) / 2
        main_plot_left = (1 - main_size) / 2
        main_plot_width = main_size

        main_ax = fig.add_axes([main_plot_left, main_plot_bottom, main_plot_width, main_plot_height])
        for spine in ["top", "right", "bottom", "left",]:
            main_ax.spines[spine].set_visible(False)
        main_ax.set_xticks([])
        main_ax.set_yticks([])
        

    else:
        ax_existed = True

    main_ax.scatter(
        latents_embedded[:, 0],
        latents_embedded[:, 1],
        alpha=0.01,
        color="darkgray",
        s=1,
    )

    n_plots = 2
    height = 1 / n_plots * main_size
    width = main_plot_left

    shrink = 0.02

    arrows = {
        45: (30, 20, 10, 5),
        135: (-30, 20, -10, 5),
        225: (-30, -20, -10, -5),
        315: (30, -20, 10, -5),
    }
    
    i_ax = 1
    for left, colormap_names in (
        (
            main_plot_left + main_size,
            {
                45: "Oranges",
                315: "Purples",
            },
        ),
        (
            shrink,
            {
                135: "Greens",
                225: "Blues",
            },
        ),
    ):
        for i_angle, angle_degrees in enumerate(colormap_names.keys()):
            bottom = (len(colormap_names) - i_angle - 1) / n_plots * main_size + main_plot_bottom
            if ax_existed:
                ax = fig_ax[i_ax]
                i_ax += 1
            else:
                ax = fig.add_axes([left, bottom, width - shrink, height - shrink])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    
            points = []
            angle = angle_degrees / 360 * np.pi * 2
            if angle_degrees == 0:
                i_start = 0
            else:
                i_start = 1
            for i in range(i_start, n_points):
                radius = radius_max * i / (n_points - 1)
                points.append(np.array([np.cos(angle), np.sin(angle)]) * radius)
    
            for i, (latent_x, latent_y) in enumerate(points):
                idx = np.argmin(np.abs(latents_embedded - np.array([latent_x, latent_y])).sum(axis=1))
                main_ax.scatter(
                    latents_embedded[idx, 0],
                    latents_embedded[idx, 1],
                    alpha=1,
                    color=COLOR1,
                    s=marker_size,
                    edgecolors="k",
                )
                # ax.plot(
                #     y[idx],
                #     color=COLOR1,
                #     lw=10,
                #     alpha=0.5,
                # )
                recon = reconstruct(y[idx].reshape(1, SEQ_LEN,))
                # ax.plot(
                #     recon[0],
                #     color=COLOR2,
                #     lw=6,
                #     alpha=1,
                # )

                panel_examples(
                    a=ax,
                    y_true=y[idx],
                    y_recon=recon[0],
                    text=None,
                    text_x=0,
                    text_y=0,
                    legend=False,
                )

                main_ax.annotate(
                    '',
                    xy=(latents_embedded[idx, 0] + arrows[angle_degrees][0], latents_embedded[idx, 1] + arrows[angle_degrees][1],),
                    xytext=(latents_embedded[idx, 0] + arrows[angle_degrees][2], latents_embedded[idx, 1] + arrows[angle_degrees][3],),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                )

    if not ax_existed:
        plt.savefig(
            fname=DIR_SRC 
            / "results" 
            / "figs"
            / "model" 
            / f"twod_visualization.png",
        )
        plt.show()
        plt.close()


def main(
    model,
    ax=None,
):
    data, _ = get_data()

    try:
        print("Trying to open t-SNE Model")
        with open(
            DIR_DATA
            / "tsne_model.joblib",
            "rb",
        ) as fp:
            tsne_model = joblib.load(fp)
    except Exception as e:
        print("Error loading tsne_model", e)
        print("Fitting new model")
        latents = get_latents(
            z=data["y"],
            model=model,
        )
        tsne_model = TSNE().fit(latents)
        with open(
            DIR_DATA
            / "tsne_model.joblib",
            "wb",
        ) as fp:
            joblib.dump(tsne_model, fp, protocol=5)
