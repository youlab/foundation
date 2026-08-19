import logging
import pickle

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    r2_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def memory_efficient_prediction(
    x_train,
    x_test,
    tgt_train,
    prediction_model_cache_dir,
    name_suffix,
    cross_val,
    input_type,
    max_depth,
    classify=False,
):
    logger.info(f"Performing memory efficient prediction for {input_type}_{name_suffix}_{cross_val}_{x_train.shape[0]}")
    if classify:
        clf = ExtraTreesClassifier(
            random_state=42,
            max_depth=max_depth,
        )

        logger.info(f"Fitting classifier on x_train {x_train.shape} and tgt_train {tgt_train.shape}")
        clf.fit(
            x_train,
            tgt_train,
        )

        logger.info(f"Saving model to {prediction_model_cache_dir}")
        if prediction_model_cache_dir is not None:
            with open(
                prediction_model_cache_dir
                / f"clf_{input_type}_{name_suffix}_{cross_val}_{x_train.shape[0]}.pkl",
                "wb",
            ) as fp:
                pickle.dump(
                    obj=clf,
                    file=fp,
                )

        return clf.predict(x_test)

    regr = ExtraTreesRegressor(
        random_state=42,
        max_depth=max_depth,
    )

    logger.info(f"Fitting regression model on x_train {x_train.shape} and tgt_train {tgt_train.shape}")
    regr.fit(
        x_train,
        tgt_train,
    )

    logger.info(f"Saving model to {prediction_model_cache_dir}")
    if prediction_model_cache_dir is not None:
        pass
        # regression_filename = f"regr_{input_type}_{name_suffix}_{cross_val}_{x_train.shape[0]}.pkl"
        # full_path = prediction_model_cache_dir / regression_filename
        
        # logger.info(f"=== REGRESSION MODEL SAVE DEBUG ===")
        # logger.info(f"Filename pattern: regr_{{input_type}}_{{name_suffix}}_{{cross_val}}_{{train_size}}.pkl")
        # logger.info(f"  input_type: {input_type}")
        # logger.info(f"  name_suffix: {name_suffix}")
        # logger.info(f"  cross_val: {cross_val}")
        # logger.info(f"  train_size: {x_train.shape[0]}")
        # logger.info(f"Full filename: {regression_filename}")
        # logger.info(f"Full path: {full_path}")
        # logger.info(f"Directory exists: {prediction_model_cache_dir.exists()}")
        
        # try:
        #     with open(full_path, "wb") as fp:
        #         pickle.dump(obj=regr, file=fp)
            
        #     # Verify the file was created
        #     if full_path.exists():
        #         file_size = full_path.stat().st_size
        #         logger.info(f"✓ SUCCESS: Regression model saved successfully (size: {file_size} bytes)")
        #     else:
        #         logger.error(f"✗ FAILURE: File was not created despite no exception")
                
        # except Exception as e:
        #     logger.error(f"✗ FAILURE: Error saving regression model: {e}")
        #     logger.error(f"Exception type: {type(e).__name__}")
        #     raise
        
        # logger.info(f"=== END REGRESSION MODEL SAVE DEBUG ===")
    else:
        logger.warning("prediction_model_cache_dir is None - regression model NOT saved!")

    return regr.predict(x_test)


def train_classifiers(
    x_raw_train,
    x_latent_train,
    tgt_train,
    x_latent_fine_tuned_train,
    x_latent_end2end_train,
):
    clf_raw = ExtraTreesClassifier(random_state=42)
    clf_latent = ExtraTreesClassifier(random_state=42)
    clf_raw.fit(
        x_raw_train,
        tgt_train,
    )
    clf_latent.fit(
        x_latent_train,
        tgt_train,
    )
    if x_latent_fine_tuned_train is not None:
        clf_latent_fine_tuned = ExtraTreesClassifier(random_state=42)
        clf_latent_fine_tuned.fit(
            x_latent_fine_tuned_train,
            tgt_train,
        )
        clf_latent_end2end = ExtraTreesClassifier(random_state=42)
        clf_latent_end2end.fit(
            x_latent_end2end_train,
            tgt_train,
        )
    else:
        clf_latent_fine_tuned = None
        clf_latent_end2end = None
    return (
        clf_raw,
        clf_latent,
        clf_latent_fine_tuned,
        clf_latent_end2end,
    )


def calc_accuracy(
    y_true,
    y_pred,
    classify,
    detailed_binary=False,
):
    if detailed_binary:
        assert np.unique(y_true).size <= 2
        n_samples = y_true.shape[0]
        true_positives = ((y_true == 1) & (y_pred == 1)).sum()
        true_negatives = ((y_true == 0) & (y_pred == 0)).sum()
        false_positives = ((y_true == 0) & (y_pred == 1)).sum()
        false_negatives = ((y_true == 1) & (y_pred == 0)).sum()
        accuracy = (true_positives + true_negatives) / n_samples
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        elif y_true.sum() > 0:
            precision = 0
        else:
            precision = 1

        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        elif y_true.sum() > 0:
            recall = 0
        else:
            recall = 1
        
        if (precision + recall) > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
    return accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
    ) if classify else r2_score(
        y_true=y_true,
        y_pred=y_pred,
    )


def predict(
    x_raw_test,
    x_latent_test,
    x_latent_fine_tuned_test,
    x_latent_end2end_test,
    model_raw,
    model_latent,
    model_latent_fine_tuned,
    model_latent_end2end,
):
    if model_latent_fine_tuned is not None:
        return (
            model_raw.predict(x_raw_test),
            model_latent.predict(x_latent_test),
            model_latent_fine_tuned.predict(x_latent_fine_tuned_test),
            model_latent_end2end.predict(x_latent_end2end_test),
        )
    return (
        model_raw.predict(x_raw_test),
        model_latent.predict(x_latent_test),
        None,
        None,
    )


def loop_for_results(
    x_raw_train_original,
    x_raw_test,
    x_latent_train_original,
    x_latent_test,
    tgt_train_original,
    tgt_test,
    x_latent_fine_tuned_train_original=None,
    x_latent_fine_tuned_test=None,
    x_latent_end2end_train_original=None,
    x_latent_end2end_test=None,
    classify=True,
    train_sizes=None,
    return_predictions=False,
    return_regressors=False,
    detailed_binary=False,
    prediction_model_cache_dir=None,
    name_suffix=None,
    cross_val=-1,
    max_depth=None,
):
    if train_sizes is None:
        train_sizes = [None]
    results = {}
    
    for train_size in train_sizes:
        if train_size == 0.8:
            x_raw_train = x_raw_train_original
            x_latent_train = x_latent_train_original
            tgt_train = tgt_train_original
        else:
            (
                x_raw_train,
                _,
                x_latent_train,
                _,
                tgt_train,
                _,
            ) = train_test_split(
                x_raw_train_original,
                x_latent_train_original,
                tgt_train_original,
                train_size=train_size / 0.8,
                random_state=42,
            )

        tgt_test_raw_pred = memory_efficient_prediction(
            x_train=x_raw_train,
            x_test=x_raw_test,
            tgt_train=tgt_train,
            prediction_model_cache_dir=prediction_model_cache_dir,
            name_suffix=name_suffix,
            cross_val=cross_val,
            input_type="raw",
            max_depth=max_depth,
            classify=classify,
        )
        tgt_test_latent_pred = memory_efficient_prediction(
            x_train=x_latent_train,
            x_test=x_latent_test,
            tgt_train=tgt_train,
            prediction_model_cache_dir=prediction_model_cache_dir,
            name_suffix=name_suffix,
            cross_val=cross_val,
            input_type="latent",
            max_depth=max_depth,
            classify=classify,
        )

        raw_regressor = None
        latent_regressor = None

        if return_regressors:

            from sklearn.ensemble import ExtraTreesRegressor
            raw_regressor = ExtraTreesRegressor(
                random_state=42,
                max_depth=max_depth,
            )
            raw_regressor.fit(
                x_raw_train,
                tgt_train,
            )
            latent_regressor = ExtraTreesRegressor(
                random_state=42,
                max_depth=max_depth,
            )
            latent_regressor.fit(
                x_latent_train,
                tgt_train,
            )
        
        tgt_test_latent_fine_tuned_pred = None
        tgt_test_latent_end2end_pred = None

        if (
            tgt_test.ndim == 2
        ) and (
            tgt_test.shape[1] > 1
        ):
            results[x_raw_train.shape[0]] = {}
            for i in range(tgt_test.shape[1]):
                results[x_raw_train.shape[0]][f"raw_accuracy_{i}"] = calc_accuracy(
                            y_true=tgt_test[:, i],
                            y_pred=tgt_test_raw_pred[:, i],
                            classify=classify,
                            detailed_binary=detailed_binary,
                        )
                results[x_raw_train.shape[0]][f"latent_accuracy_{i}"] = calc_accuracy(
                            y_true=tgt_test[:, i],
                            y_pred=tgt_test_latent_pred[:, i],
                            classify=classify,
                            detailed_binary=detailed_binary,
                        )
        else:
            results[x_raw_train.shape[0]] = {
                    "raw_accuracy": calc_accuracy(
                        y_true=tgt_test,
                        y_pred=tgt_test_raw_pred,
                        classify=classify,
                        detailed_binary=detailed_binary,
                    ),
                    "latent_accuracy": calc_accuracy(
                        y_true=tgt_test,
                        y_pred=tgt_test_latent_pred,
                        classify=classify,
                        detailed_binary=detailed_binary,
                    ),
            }

    results["test_size"] = x_raw_test.shape[0]
    # if return_predictions:
    #     return results, tgt_test, tgt_test_raw_pred, tgt_test_latent_pred, tgt_test_latent_fine_tuned_pred, tgt_test_latent_end2end_pred
    
    if return_predictions:
        return (
            results,
            tgt_test,
            tgt_test_raw_pred,
            tgt_test_latent_pred,
            tgt_test_latent_fine_tuned_pred,
            tgt_test_latent_end2end_pred,
        )

    if return_regressors:
        return (
            results,
            raw_regressor,
            latent_regressor,
        )

    return results
