import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import trange

from applications.utils.data import get_t_cols
from config import PATH_FUJITA_DATA
from figs.utils.colors import get_colors


def gpy(y):
    N_REPEATS = 5
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    
    x = np.linspace(
        0,
        127,
        y.shape[1],
        dtype=int,
    ).reshape(-1, 1)
    x_fit = np.linspace(
        0,
        128 - 1,
        128,
        dtype=int,
    ).reshape(-1, 1)

    y_new = np.zeros((
        y.shape[0],
        128,
    ))
    
    for i_row in trange(y.shape[0]):
        row_hold = np.zeros((N_REPEATS, x_fit.shape[0]))
        row = y[i_row, :].reshape(-1, 1)
        for i_repeat in range(N_REPEATS):
            gaussian_process.fit(x, row)
            row_hold[i_repeat, :] = gaussian_process.predict(x_fit).flatten()
    
        row_fit = np.median(a=row_hold, axis=0,)
        
        y_new[i_row, :] = row_fit

    return y_new, x


def format_a(
    a,
    xmax,
    ymin,
    ymax,
    yticks,
    xticks,
    ytick_labels,
    xlabel,
    ylabel,
):
    a.set_ylim(ymin, ymax)
    a.set_xlim(0, xmax)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.set_xticks(
        xticks,
        labels=xticks,
        fontsize=16,
        fontweight="medium",
    )
    a.set_yticks(
        yticks,
        labels=ytick_labels,
        fontsize=16,
        fontweight="medium",
    )
    a.set_xlabel(
        xlabel,
        fontsize=18,
        fontweight="medium",
    )
    a.set_ylabel(
        ylabel,
        fontsize=18,
        fontweight="medium",
    )


def main(ax):
    df = pd.read_csv(PATH_FUJITA_DATA)
    t_cols, _ = get_t_cols(p=df)
    x = df.loc[:, t_cols].to_numpy()

    n_samples = 10
    x = x[:n_samples]
    x16 = x[:, 40:56]
    x16 = (x16 / x16.sum(axis=0).reshape(1, -1))

    x = x[x16.max(axis=1) > 0]
    x16 = x16[x16.max(axis=1) > 0]

    n_samples = x.shape[0]
    colors = get_colors(n=n_samples)
    x_norm = x16 / x16.max(axis=1).reshape(-1, 1)
    x_interp, refx = gpy(y=x_norm)

    alpha = 0.8

    for i in range(n_samples):
        ax[0].plot(
            x[i],
            marker="o",
            color=colors[i],
            alpha=alpha,
            lw=3,
        )
        ax[1].plot(
            x16[i],
            marker="o",
            color=colors[i],
            alpha=alpha,
            lw=3,
        )
        ax[2].plot(
            x_interp[i],
            color=colors[i],
            alpha=alpha,
            lw=3,
        )
        ax[2].scatter(
            refx,
            x_norm[i],
            marker="o",
            color=colors[i],
            alpha=alpha,
        )

    format_a(
        a=ax[0],
        xmax=110,
        ymin=0,
        ymax=x.max() * 1.05,
        yticks=[0, 10000],
        ytick_labels=[0, "10,000",],
        xticks=[0, 50, 100],
        xlabel="Days",
        ylabel="Seq. reads",
    )

    format_a(
        a=ax[1],
        xmax=16,
        ymin=0,
        ymax=x16.max() * 1.05,
        yticks=[0, 1],
        ytick_labels=[0, 1,],
        xticks=[0, 5, 10, 15],
        xlabel="Days",
        ylabel="Relative fraction",
    )
    format_a(
        a=ax[2],
        xmax=128,
        ymin=0,
        ymax=1.05,
        yticks=[0, 1],
        ytick_labels=[0, 1],
        xticks=[0, 50, 100],
        xlabel="Time points",
        ylabel="Normalized value",
    )
