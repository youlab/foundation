import logging

import numpy as np

from applications.config import NOW_TEXT
from applications.consortia.future_one_by_one import main as main_future_plus_one
from applications.consortia.future_outlook import main as main_future_outlook
from applications.consortia_exp.config import get_config
from applications.datasets.future_simulation_dataset import FutureSimulationDataset
from applications.datasets.raw_fujita_dataset import RawDataset
from ml.utils.load_models import load_default_model
from config import (
    DIR_CACHE_CONSORTIA_EXP,
    MODEL_TYPE,
    Z_DIM,
)

logger = logging.getLogger(__name__)


def get_data(
    test_rep,
    model,
    use_raw,
    source_file,
    strains,
    step_size,
    tgt_type,
):
    raw_data = RawDataset(
        test_reps=[test_rep],
        source_file=source_file,
        strains=strains,
        interp_mult=8,
    )
    dataset_train = FutureSimulationDataset(
        y=raw_data.y_train,
        n_focal=raw_data.n_focal,
        train_or_test="train",
        train_size=1,
        tgt_type=tgt_type,
        model=model,
        model_type=MODEL_TYPE,
        z_dim=Z_DIM,
        use_raw=use_raw,
        step_size=step_size,
        save_cache=False,
    )

    dataset_test = FutureSimulationDataset(
        y=raw_data.y_test,
        n_focal=raw_data.n_focal,
        train_or_test="test",
        train_size=1,
        tgt_type=tgt_type,
        model=model,
        model_type=MODEL_TYPE,
        z_dim=Z_DIM,
        use_raw=use_raw,
        step_size=step_size,
        save_cache=False,
    )
    logger.info(f"rd train {raw_data.y_train.shape} rd test {raw_data.y_test.shape} dtrain {dataset_train.x_raw.shape}  {dataset_train.tgt_raw.shape} dtest {dataset_test.x_raw.shape} {dataset_test.tgt_raw.shape} my test {np.concatenate((dataset_test.x_raw, dataset_test.tgt_raw), axis=2).shape}")
    return raw_data, dataset_train, dataset_test


def get_focus_params(task_id=None):
    from itertools import product

    source_files = [
        "soil-a",
        "soil-b",
        "soil-c",
        "water-a",
        "water-b",
        "water-c",
    ]
    reps = np.arange(8)
    input_types = ["raw", "latent",]
    tgt_types = ["segment128", "latent",]
    params = list(product(source_files, reps, input_types, tgt_types,))
    if task_id is None:
        return params
    return params[task_id]


def main(
    task_id,
):
    model = load_default_model()

    (
        source_file,
        rep_idx,
        input_type,
        tgt_type,
    ) = get_focus_params(task_id=task_id)
    if (
        input_type == "raw"
    ) and (
        tgt_type == "latent"
    ):
        return
    if (
        input_type == "latent"
    ) and (
        tgt_type == "segment128"
    ):
        return
    (
        strains,
        test_reps,
    ) = get_config(source_file=source_file)
    test_rep = test_reps[rep_idx]

    name_suffix = f"{source_file}-{test_rep}_{input_type}_{tgt_type}"
    logger.info(name_suffix)
    raw_data, dataset_train, dataset_test = get_data(
        test_rep=test_rep,
        model=model,
        use_raw=input_type == "raw",
        source_file=source_file,
        strains=strains,
        step_size=1,
        tgt_type=tgt_type,
    )
    try:
        np.load(
            DIR_CACHE_CONSORTIA_EXP
            / f"regr_pred_{name_suffix}_{NOW_TEXT}.npz",
        )
    except Exception as e:
        logger.warning(f"Could not load 'regr_pred_{name_suffix}_{NOW_TEXT}.npz': {e}")
        main_future_plus_one(
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            dir_cache=DIR_CACHE_CONSORTIA_EXP,
            name=source_file,
            name_suffix=name_suffix,
        )
    try:
        np.load(
            DIR_CACHE_CONSORTIA_EXP
            / f"future_outlook_{name_suffix}_{NOW_TEXT}.npz",
        )
    except Exception as e:
        logger.warning(f"Could not load 'regr_pred_{name_suffix}_{NOW_TEXT}.npz': {e}")
        main_future_outlook(
            raw_data=raw_data,
            use_raw=input_type == "raw",
            tgt_type=tgt_type,
            dir_cache=DIR_CACHE_CONSORTIA_EXP,
            name=source_file,
            name_suffix=name_suffix,
            model=model,
            model_type=MODEL_TYPE,
            z_dim=Z_DIM,
        )
