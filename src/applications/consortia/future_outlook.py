import logging
import pickle

import numpy as np

from config import SEQ_LEN
from applications.config import NOW_TEXT
from applications.utils.latents import get_latents, decode

logger = logging.getLogger(__name__)


def get_predictions(
    w,
    n_focal,
    regr,
    use_raw,
    tgt_type,
    model,
    model_type,
    z_dim,
):
    w_pred = np.zeros_like(w)
    w_pred[:, :, :SEQ_LEN] = w[:, :, :SEQ_LEN]

    if (
        tgt_type.find("segment") > -1
    ) or (
        tgt_type.find("sliders") > -1
    ):
        step = int(tgt_type[7:])
    elif tgt_type == "latent":
        step = SEQ_LEN
    else:
        step = 1
    for i in range(SEQ_LEN, w_pred.shape[2], step):
        assert w_pred[:, :, i].sum() == 0
        x_in = w_pred[:, :, i - SEQ_LEN : i]
        x_in = x_in.reshape(-1, SEQ_LEN)
        if not use_raw:
            x_in = get_latents(
                z=x_in,
                use_transformer=model_type == "MCR",
                z_dim=z_dim,
                model=model,
            )
            x_in = x_in.reshape(-1, n_focal * (z_dim + 1))
        else:
            x_in = x_in.reshape(-1, n_focal * SEQ_LEN)
        x_in[np.isnan(x_in)] = 0
        pred = regr.predict(x_in)
        if tgt_type == "nextval":
            pass
        elif tgt_type == "log":
            pred = 10 ** pred
        elif tgt_type == "dydt":
            pred = w_pred[:, :, i-1] + w_pred[:, :, i-1] * pred / 100
        elif tgt_type.find("segment") > -1:
            pred = pred.reshape(-1, step)
        elif tgt_type == "latent":
            pred = decode(
                z=pred.reshape(-1, 1, z_dim + 1),
                model=model,
            )
            pred.reshape(-1, SEQ_LEN,)
        elif tgt_type.find("slide") > -1:
            pred = decode(
                z=pred.reshape(-1, 1, z_dim + 1),
                model=model,
            )
            pred.reshape(-1, SEQ_LEN,)
            pred = pred[:, -step:]

        pred[pred < 0] = 0
        pred[pred > 1] = 1
        pred = pred.reshape(w_pred.shape[0], w_pred.shape[1], step)
        if step > 1:
            w_pred[:, :, i:i+step] = pred
        else:
            w_pred[:, :, i] = pred
    return w_pred


def main(
    raw_data,
    use_raw,
    tgt_type,
    dir_cache,
    name,
    name_suffix,
    model,
    model_type,
    z_dim,
):
    logger.info(f"Opening regression model for {name}")
    
    expected_filename = f"regr_{name_suffix}.pkl"
    full_path = dir_cache / expected_filename
    
    logger.info(f"=== REGRESSION MODEL LOAD DEBUG ===")
    logger.info(f"Expected filename: {expected_filename}")
    logger.info(f"Full path: {full_path}")
    logger.info(f"Cache directory: {dir_cache}")
    logger.info(f"File exists: {full_path.exists()}")
    
    # List all regression files in cache to help debugging
    if dir_cache.exists():
        regr_files = list(dir_cache.glob("regr_*.pkl"))
        logger.info(f"Available regression files ({len(regr_files)}):")
        for f in sorted(regr_files):
            logger.info(f"  - {f.name}")
    else:
        logger.error(f"Cache directory does not exist: {dir_cache}")
    
    try:
        with open(full_path, "rb") as fp:
            regr = pickle.load(fp)
        logger.info(f"✓ SUCCESS: Regression model loaded successfully")
        logger.info(f"=== END REGRESSION MODEL LOAD DEBUG ===")
    except FileNotFoundError as e:
        logger.error(f"✗ FAILURE: Regression model file not found")
        logger.error(f"This might indicate:")
        logger.error(f"  1. RUN_CONSORTIA_SIM_V2 was not run for the required task_id")
        logger.error(f"  2. File naming mismatch between save and load operations")
        logger.error(f"  3. Files were saved to a different directory")
        logger.error(f"=== END REGRESSION MODEL LOAD DEBUG ===")
        raise
    except Exception as e:
        logger.error(f"✗ FAILURE: Error loading regression model: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"=== END REGRESSION MODEL LOAD DEBUG ===")
        raise

    logger.info(f"Building data for {name}")
    w_train = raw_data.y_train.reshape(
        -1,
        raw_data.n_focal,
        raw_data.y_train.shape[1],
    )
    w_test = raw_data.y_test.reshape(
        -1,
        raw_data.n_focal,
        raw_data.y_test.shape[1],
    )

    logger.info(f"Making predictions for {name}")
    w_train_pred = get_predictions(
        w=w_train,
        n_focal=raw_data.n_focal,
        regr=regr,
        use_raw=use_raw,
        tgt_type=tgt_type,
        model=model,
        model_type=model_type,
        z_dim=z_dim,
    )
    w_test_pred = get_predictions(
        w=w_test,
        n_focal=raw_data.n_focal,
        regr=regr,
        use_raw=use_raw,
        tgt_type=tgt_type,
        model=model,
        model_type=model_type,
        z_dim=z_dim,
    )

    logger.info(f"Saving data for {name}")
    np.savez(
        dir_cache
        / f"future_outlook_{name_suffix}_{NOW_TEXT}.npz",
        w_train=w_train,
        w_test=w_test,
        w_train_pred=w_train_pred,
        w_test_pred=w_test_pred,
    )
    logger.info(f"Completed {name}")
