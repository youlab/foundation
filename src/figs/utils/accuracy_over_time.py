import numpy as np

from config import SEQ_LEN


def plot_r2(
    a,
    data,
    color,
    ls,
    lw=4,
    alpha=0.7,
):
    a.plot(
        np.arange(SEQ_LEN, data.size + SEQ_LEN,),
        data,
        color=color,
        ls=ls,
        lw=lw,
        alpha=alpha,
        label=r"latent $R^2$" if ls == "-" else r"raw $R^2$",
    )


def plot_rmse(
    a,
    data,
    color,
    ls,
    lw=4,
    alpha=0.7,
):
    a.plot(
        np.arange(SEQ_LEN, data.size + SEQ_LEN,),
        data,
        color=color,
        ls=ls,
        lw=lw,
        alpha=alpha,
        label="latent RMSE" if ls == "-" else "raw RMSE",
    )
