import numpy as np
import pandas as pd

from applications.antibiotics.helena.inputs import (
    get_inputs,
    get_train_and_test_sets,
)
from applications.utils.models import (
    fit_models,
    predict,
    calc_metrics,
)
from applications.utils.plots import (
    plot_train_ratio_comparison,
    plot_results,
)
from config import DIR_CACHE_HELENA


def main():
    x, y_all, y_class, y_conc, encoder = get_inputs()
    
    results = []
    for train_ratio in [1., 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,]:

        for input_type, inputs in [("full_curve", y_all), ("latents", x,),]:
            print(f"Working on {input_type} with train_ratio {train_ratio}")

            x_train, x_test, y_class_train, y_class_test, y_conc_train, y_conc_test = get_train_and_test_sets(
                x=inputs,
                y_class=y_class,
                y_conc=y_conc,
                train_ratio=0.8,
            )

            if train_ratio != 1.0:
                x_train, _, y_class_train, _, y_conc_train, _ = get_train_and_test_sets(
                    x=x_train,
                    y_class=y_class_train,
                    y_conc=y_conc_train,
                    train_ratio=train_ratio,
                )

            regr, clf = fit_models(
                x_train=x_train,
                y_regr_train=y_conc_train,
                y_class_train=y_class_train,
            )

            y_conc_train_pred, y_conc_test_pred, y_class_train_pred, y_class_test_pred = predict(
                x_train=x_train,
                x_test=x_test,
                regr=regr,
                clf=clf,
            )

            r2_train, r2_test, accuracy_train, accuracy_test = calc_metrics(
                    y_regr_train=y_conc_train,
                    y_regr_train_pred=y_conc_train_pred,
                    y_regr_test=y_conc_test,
                    y_regr_test_pred=y_conc_test_pred,
                    y_class_train=y_class_train,
                    y_class_train_pred=y_class_train_pred,
                    y_class_test=y_class_test,
                    y_class_test_pred=y_class_test_pred,
            )

            results.append({
                "input_type": input_type,
                "train_ratio": train_ratio * 0.8,
                "accuracy_train": accuracy_train,
                "accuracy_test": accuracy_test,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "train_size": x_train.shape[0],
                "test_size": x_test.shape[0],
            })

            plot_results(
                y_class_train=y_class_train,
                y_class_train_pred=y_class_train_pred,
                y_class_test=y_class_test,
                y_class_test_pred=y_class_test_pred,
                y_regr_train=y_conc_train,
                y_regr_train_pred=y_conc_train_pred,
                y_regr_test=y_conc_test,
                y_regr_test_pred=y_conc_test_pred,
                labels=encoder.categories_[0],
                file_path=DIR_CACHE_HELENA,
                file_name=f"helena_{input_type}_{str(train_ratio * 0.8).replace('.', 'p')}",
            )

    df = pd.DataFrame.from_records(results)
    df.to_csv(DIR_CACHE_HELENA / "helena_results.csv")
    plot_train_ratio_comparison(
        file_path=DIR_CACHE_HELENA,
        file_name="helena_results",
    )


def plasmids():
    x, y_all, y_class, y_conc, encoder, y_plasmids, encoder_plasmids = get_inputs(return_plasmids=True)
    
    labels_plasmids = encoder_plasmids.categories_[0]
    labels_plasmids[-1] = "none"
    y_plasmids[np.isnan(y_plasmids)] = len(labels_plasmids) - 1

    x = x[:, :9]
    y_all = y_all[:, :128]

    results = []
    for train_ratio in [1., 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,]:

        for input_type, inputs in [("full_curve", y_all), ("latents", x,),]:
            print(f"Working on {input_type} with train_ratio {train_ratio}")

            x_train, x_test, y_class_train, y_class_test, y_conc_train, y_conc_test = get_train_and_test_sets(
                x=inputs,
                y_class=y_plasmids,
                y_conc=y_conc,
                train_ratio=0.8,
            )

            if train_ratio != 1.0:
                x_train, _, y_class_train, _, y_conc_train, _ = get_train_and_test_sets(
                    x=x_train,
                    y_class=y_class_train,
                    y_conc=y_conc_train,
                    train_ratio=train_ratio,
                )

            regr, clf = fit_models(
                x_train=x_train,
                y_regr_train=y_conc_train,
                y_class_train=y_class_train,
            )

            y_conc_train_pred, y_conc_test_pred, y_class_train_pred, y_class_test_pred = predict(
                x_train=x_train,
                x_test=x_test,
                regr=regr,
                clf=clf,
            )

            r2_train, r2_test, accuracy_train, accuracy_test = calc_metrics(
                    y_regr_train=y_conc_train,
                    y_regr_train_pred=y_conc_train_pred,
                    y_regr_test=y_conc_test,
                    y_regr_test_pred=y_conc_test_pred,
                    y_class_train=y_class_train,
                    y_class_train_pred=y_class_train_pred,
                    y_class_test=y_class_test,
                    y_class_test_pred=y_class_test_pred,
            )

            results.append({
                "input_type": input_type,
                "train_ratio": train_ratio * 0.8,
                "accuracy_train": accuracy_train,
                "accuracy_test": accuracy_test,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "train_size": x_train.shape[0],
                "test_size": x_test.shape[0],
            })

            plot_results(
                y_class_train=y_class_train,
                y_class_train_pred=y_class_train_pred,
                y_class_test=y_class_test,
                y_class_test_pred=y_class_test_pred,
                y_regr_train=y_conc_train,
                y_regr_train_pred=y_conc_train_pred,
                y_regr_test=y_conc_test,
                y_regr_test_pred=y_conc_test_pred,
                labels=labels_plasmids,
                file_path=DIR_CACHE_HELENA,
                file_name=f"helena_plasmids_{input_type}_{str(train_ratio * 0.8).replace('.', 'p')}",
            )

    df = pd.DataFrame.from_records(results)
    df.to_csv(DIR_CACHE_HELENA / "helena_plasmids_results.csv")
    plot_train_ratio_comparison(
        file_path=DIR_CACHE_HELENA,
        file_name="helena_plasmids_results",
    )
