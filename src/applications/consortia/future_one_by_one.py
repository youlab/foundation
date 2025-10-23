import logging
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from applications.config import NOW_TEXT

logger = logging.getLogger(__name__)

def train_regressor(
    x_train,
    x_test,
    tgt_train,
    dir_cache,
    file_name,
):
    regr = ExtraTreesRegressor(
        random_state=42,
        criterion="friedman_mse",
        max_depth=15,
    )

    regr.fit(
        x_train,
        tgt_train,
    )

    with open(
        dir_cache
        / file_name,
        "wb",
    ) as fp:
        pickle.dump(
            obj=regr,
            file=fp,
        )

    tgt_train_pred = regr.predict(x_train)
    tgt_test_pred = regr.predict(x_test)

    return tgt_train_pred, tgt_test_pred


def main(
    dataset_train,
    dataset_test,
    dir_cache,
    name,
    name_suffix,
):
    # Prepare data for training the regression models
    logger.info(f"{name} getting regression inputs")
    x_train = dataset_train.x.reshape(dataset_train.x.shape[0], -1)
    x_test = dataset_test.x.reshape(dataset_test.x.shape[0], -1)
    tgt_train = dataset_train.tgt.reshape(dataset_train.tgt.shape[0], -1)
    tgt_test = dataset_test.tgt.reshape(dataset_test.tgt.shape[0], -1)
    tgt_raw_train = dataset_train.tgt_raw.reshape(dataset_train.tgt_raw.shape[0], -1)
    tgt_raw_test = dataset_test.tgt_raw.reshape(dataset_test.tgt_raw.shape[0], -1)

    x_train[np.isnan(x_train)] = 0
    x_test[np.isnan(x_test)] = 0
    tgt_train[np.isnan(tgt_train)] = 0
    tgt_test[np.isnan(tgt_test)] = 0
    
    
    logger.info(f"{name} training regressor, {x_train.shape}, {tgt_train.shape}")
    (
        tgt_train_pred,
        tgt_test_pred,
    ) = train_regressor(
        x_train=x_train,
        x_test=x_test,
        tgt_train=tgt_train,
        dir_cache=dir_cache,
        file_name=f"regr_{name_suffix}.pkl",
    )

    logger.info(f"{name} saving data")
    np.savez(
        file=dir_cache / f"regr_pred_{name_suffix}_{NOW_TEXT}.npz",
        x_train=x_train,
        x_test=x_test,
        tgt_train=tgt_train,
        tgt_test=tgt_test,
        tgt_train_pred=tgt_train_pred,
        tgt_test_pred=tgt_test_pred,
        tgt_raw_train=tgt_raw_train,
        tgt_raw_test=tgt_raw_test,
    )

    logger.info(f"{name} complete")
