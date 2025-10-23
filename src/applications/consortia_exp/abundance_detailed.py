import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from tqdm import tqdm

from applications.config import NOW_TEXT
from applications.consortia_exp.config import get_config
from applications.utils.latents import get_latents
from ml.utils.load_models import load_default_model
from config import (
    DIR_SRC,
    DIR_RESULTS_CONSORTIA_EXP,
)
from applications.antibiotics.loop_for_results import calc_accuracy


def train_classifier(
    x_train,
    x_test,
    tgt_train,
):
    clf = ExtraTreesClassifier(
        random_state=42,
    )

    clf.fit(
        x_train,
        tgt_train,
    )

    tgt_test_pred = clf.predict(x_test)

    return tgt_test_pred


def train_regressor(
    x_train,
    x_test,
    tgt_train,
):
    regr = ExtraTreesRegressor(
        random_state=42,
    )

    regr.fit(
        x_train,
        tgt_train,
    )

    tgt_test_pred = regr.predict(x_test)

    return tgt_test_pred


def main(
    points_aheads=[10, 20, 40, 80, 160, 320, 640,],
    threshold=5,
    classify=True,
):
    model = load_default_model()
    source_files = [
        "soil-a",
        "soil-b",
        "soil-c",
        "water-a",
        "water-b",
        "water-c",
    ]
    results = []
    for points_ahead in tqdm(points_aheads, "points ahead"):
        for source_file in tqdm(source_files, "source files",):
            _, test_reps = get_config(source_file=source_file)
            for input_type, tgt_type in [
                ("raw", "segment128",),
                ("latent", "latent",),
            ]:
                final_true = []
                final_pred = []
                for test_rep in test_reps:
                    name_suffix = f"{source_file}-{test_rep}_{input_type}_{tgt_type}"
                    future_outlook = np.load(
                        DIR_RESULTS_CONSORTIA_EXP
                        / f"future_outlook_{name_suffix}_{NOW_TEXT}.npz",
                    )
                    if input_type == "raw":
                        x_train = future_outlook["w_train"][:, :, :128].reshape(future_outlook["w_train"].shape[0], -1)
                        x_test = future_outlook["w_test"][:, :, :128].reshape(future_outlook["w_test"].shape[0], -1)
                    else:
                        x_train = get_latents(
                            z=future_outlook["w_train"][:, :, :128].reshape(-1, 128),
                            model=model,
                        ).reshape(future_outlook["w_train"].shape[0], -1)
                        x_test = get_latents(
                            z=future_outlook["w_test"][:, :, :128].reshape(-1, 128),
                            model=model,
                        ).reshape(future_outlook["w_test"].shape[0], -1)
        
                    tgt_train = future_outlook["w_train"][:, :, 128 + points_ahead:128 + points_ahead + 10].mean(axis=2)
                    tgt_test = future_outlook["w_test"][:, :, 128 + points_ahead:128 + points_ahead + 10].mean(axis=2)
                    if classify:
                        tgt_train = (tgt_train > (threshold / 100)) * 1
                        tgt_test = (tgt_test > (threshold / 100)) * 1
                        tgt_test_pred = train_classifier(
                            x_train=x_train,
                            x_test=x_test,
                            tgt_train=tgt_train,
                        )
                    else:
                        tgt_train = np.around(tgt_train, 1)
                        tgt_test = np.around(tgt_test, 1)
                        tgt_test_pred = train_regressor(
                            x_train=x_train,
                            x_test=x_test,
                            tgt_train=tgt_train,
                        )
                    final_true.append(tgt_test.flatten())
                    final_pred.append(tgt_test_pred.flatten())
                final_true = np.concatenate(final_true).flatten()
                final_pred = np.concatenate(final_pred).flatten()
                accuracy = calc_accuracy(
                    y_true=final_true,
                    y_pred=final_pred,
                    classify=classify,
                    detailed_binary=False,
                )
                results.append(
                    {
                        "source_file": source_file,
                        
                        "input_type": input_type,
                        "accuracy": accuracy,
                        "points_ahead": points_ahead,
                    }
                )

    df = pd.DataFrame.from_records(results)

    fig, ax = plt.subplots(2, 3, figsize=(8, 5,), sharex=True, sharey=True,)
    ax = ax.ravel()

    for i_ax, source_file in enumerate(source_files):
        mask_sf = df.source_file == source_file
        for input_type in ["raw", "latent",]:
            ax[i_ax].plot(
                df.loc[mask_sf & (df.input_type == input_type), "points_ahead"].to_numpy(),
                df.loc[mask_sf & (df.input_type == input_type), "accuracy"].to_numpy(),
            )
    ax[0].set_ylim(-0.1, 1.1)
    plt.savefig(DIR_RESULTS_CONSORTIA_EXP / "abundance_task_by_source_file.png")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5,), sharex=True, sharey=True,)

    for points_ahead in df.points_ahead.unique():
        mask_pa = df.points_ahead == points_ahead
        for input_type in ["raw", "latent",]:
            ax.scatter(
                df.loc[mask_pa & (df.input_type == input_type), "points_ahead"].to_numpy().mean(),
                df.loc[mask_pa & (df.input_type == input_type), "accuracy"].to_numpy().mean(),
                color="tab:blue" if (input_type == "raw") else "tab:orange",
            )
    ax.set_ylim(-0.1, 1.1)
    plt.savefig(DIR_RESULTS_CONSORTIA_EXP / "abundance_task_average.png")
    plt.close()
    df.to_csv(DIR_RESULTS_CONSORTIA_EXP / "abundance_task_df.csv")
