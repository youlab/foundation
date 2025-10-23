import numpy as np
from sklearn.metrics import r2_score

from applications.utils.latents import reconstruct
from config import DIR_CACHE_MODEL_FIGS
from data.utils import get_data
from ml.utils.load_models import load_default_model


def make_plot(
    a,
    y_true,
    y_pred,
    r2,
    color,
    dataset,
    n_samples=100_000,
    alpha=0.05,
    s=1,
    x_r2=0.2,
    y_r2=0.8,
    fs_r2=14,
    horizontal_alignment="center",
    vertical_alignment="center",
    x_dataset=0.8,
    y_dataset=0.1,
):
    step = y_true.size // n_samples
    a.scatter(
        y_true.flatten()[0::step],
        y_pred.flatten()[0::step],
        color=color,
        alpha=alpha,
        s=s,
    )
    a.plot(
        [0, 1,],
        [0, 1,],
        "k:",
        label="y=x",
    )
    a.text(
        x=x_r2,
        y=y_r2,
        s=r"$R^2=$" + f"{r2:.3f}",
        horizontalalignment=horizontal_alignment,
        verticalalignment=vertical_alignment,
        fontsize=fs_r2,
        fontweight="semibold",
    )
    a.text(
        x=x_dataset,
        y=y_dataset,
        s=dataset,
        horizontalalignment=horizontal_alignment,
        verticalalignment=vertical_alignment,
        fontsize=fs_r2,
        fontweight="semibold",
    )


def plot_model_reconstruction_accuracy(
    ax,
    fs_ticks=14,
    fs_label=16,
    p_x=0.05,
    p_y=0.05,
    x_r2=0.1,
    y_r2=0.9,
    fs_r2=14,
    horizontal_alignment="left",
    x_dataset=0.6,
    use_cache=True,
    fw_label="medium",
):
    data, _ = get_data(category="all")
    y_train = data["y"][data["train_idx"]]
    y_test = data["y"][data["test_idx"]]
    
    if use_cache:
        try:
            reconstruction_train = np.load(
                DIR_CACHE_MODEL_FIGS
                / "reconstruction_train.npy",
            )
            reconstruction_test = np.load(
                DIR_CACHE_MODEL_FIGS
                / "reconstruction_test.npy",
            )
            r2_train, r2_test = np.load(
                DIR_CACHE_MODEL_FIGS
                / "r2s.npy",
            )
        except:
            use_cache = False

    if not use_cache:
        model = load_default_model()

        reconstruction_train = reconstruct(
            model=model,
            x=y_train,
            batch_size=4096,
            is_vae=True,
            is_pca=False,
        )
        reconstruction_test = reconstruct(
            model=model,
            x=y_test,
            batch_size=4096,
            is_vae=True,
            is_pca=False,
        )

        r2_train = r2_score(
            y_true=y_train.flatten(),
            y_pred=reconstruction_train.flatten(),
        )

        r2_test = r2_score(
            y_true=y_test.flatten(),
            y_pred=reconstruction_test.flatten(),
        )
        DIR_CACHE_MODEL_FIGS.mkdir(parents=True, exist_ok=True)
        np.save(
            DIR_CACHE_MODEL_FIGS
            / "reconstruction_train.npy",
            reconstruction_train,
        )
        np.save(
            DIR_CACHE_MODEL_FIGS
            / "reconstruction_test.npy",
            reconstruction_test,
        )
        np.save(
            DIR_CACHE_MODEL_FIGS
            / "r2s.npy",
            np.array([r2_train, r2_test]),
        )

    ax.axis("off")
    w = (1.1 - p_x * 4) / 2
    h = 1 - p_y * 2
    a = []
    for i_col in range(2):
        a.append(
            ax.inset_axes(
                [
                    p_x + (w + p_x * 2) * i_col,
                    p_y,
                    w,
                    h,
                ]
            )
        )
    ax = np.array(a)

    make_plot(
        a=ax[0],
        y_true=y_train,
        y_pred=reconstruction_train,
        r2=r2_train,
        color="tab:blue",
        dataset="Train",
        x_r2=x_r2,
        y_r2=y_r2,
        horizontal_alignment="left",
        x_dataset=x_dataset,
        fs_r2=fs_r2,
    )
    make_plot(
        a=ax[1],
        y_true=y_test,
        y_pred=reconstruction_test,
        r2=r2_test,
        color="tab:orange",
        dataset="Test",
        x_r2=x_r2,
        y_r2=y_r2,
        horizontal_alignment="left",
        x_dataset=x_dataset,
        fs_r2=fs_r2,
    )
    
    ax[0].set_xticks(
        [0, 1,],
        labels=[0, 1,],
        fontsize=fs_ticks,
    )
    ax[1].set_xticks(
        [0, 1,],
        labels=[],
        fontsize=fs_ticks,
    )
    
    ax[0].set_yticks(
        [0, 1,],
        labels=[0, 1,],
        fontsize=fs_ticks,
    )
    ax[1].set_yticks(
        [0, 1,],
        labels=[],
        fontsize=fs_ticks,
    )
    
    ax[0].set_xlabel(
        "True",
        fontsize=fs_label,
        fontweight=fw_label,
    )
    ax[0].set_ylabel(
        "Predicted",
        fontsize=fs_label,
        fontweight=fw_label,
    )
    for a in ax:
        a.set_xticks(
            [0.25, 0.5, 0.75,],
            minor=True,
        )
        a.set_yticks(
            [0.25, 0.5, 0.75,],
            minor=True,
        )
        a.set_ylim(
            0,
            1,
        )
        a.set_xlim(
            0,
            1,
        )
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)
