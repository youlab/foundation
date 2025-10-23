import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from applications.antibiotics.loop_for_results import calc_accuracy
from applications.config import NOW_TEXT
from applications.consortia_exp.config import get_config
from applications.utils.latents import get_latents
from ml.utils.load_models import load_default_model
from config import DIR_RESULTS_CONSORTIA_EXP, DIR_CACHE_CONSORTIA_EXP

logger = logging.getLogger(__name__)


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


def main(
    points_back,
    threshold=10,
):
    assert isinstance(threshold, int)
    model = load_default_model()

    tgt_type = "latent"
    source_files = [
        "soil-a",
        "soil-b",
        "soil-c",
        "water-a",
        "water-b",
        "water-c",
    ]
    results = []
    for source_file in source_files:
        _, test_reps = get_config(source_file=source_file)
        for input_type, tgt_type in [
            ("raw", "segment128",),
            ("latent", "latent",),
        ]:
            final_true = []
            final_pred = []
            for test_rep in test_reps:
                name_suffix = f"{source_file}-{test_rep}_{input_type}_{tgt_type}"
                if input_type == "raw":
                    future_outlook = np.load(
                        DIR_CACHE_CONSORTIA_EXP
                        / f"future_outlook_{name_suffix}_{NOW_TEXT}.npz",
                    )
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
                tgt_train = (future_outlook["w_train"][:, :, -1] > (threshold / 100)) * 1
                tgt_test = (future_outlook["w_test"][:, :, -1] > (threshold / 100)) * 1
                if points_back > 1:
                    tgt_train = (future_outlook["w_train"][:, :, -points_back:].mean(axis=2) > (threshold / 100)) * 1
                    tgt_test = (future_outlook["w_test"][:, :, -points_back:].mean(axis=2) > (threshold / 100)) * 1
                else:
                    tgt_train = (future_outlook["w_train"][:, :, -1] > (threshold / 100)) * 1
                    tgt_test = (future_outlook["w_test"][:, :, -1] > (threshold / 100)) * 1

                tgt_test_pred = train_classifier(
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
                classify=True,
                detailed_binary=True,
            )
            results.append(
                {
                    "source_file": source_file,
                    "input_type": input_type,
                    "accuracy": accuracy["accuracy"],
                    "precision": accuracy["precision"],
                    "recall": accuracy["recall"],
                    "f1_score": accuracy["f1_score"],
                }
            )
    pd.DataFrame.from_records(results).to_csv(
        DIR_RESULTS_CONSORTIA_EXP
        / f"abundance_{points_back}_final_abundance_classification_{threshold}.csv",
    )
