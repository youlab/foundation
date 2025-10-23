import logging
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from applications.config import NOW_TEXT
from applications.utils.metrics import (
    calc_rmse,
    get_rmse,
)
from config import (
    DIR_CACHE_CONSORTIA,
    DIR_RESULTS_CONSORTIA,
    SEQ_LEN,
)

logger = logging.getLogger(__name__)

def define_consortium(file_name):
    if (
        file_name.find("simple") > -1
    ) and (
        file_name.find("complex") == -1
    ):
        return "simple"
    if (
        file_name.find("simple") == -1
    ) and (
        file_name.find("complex") > -1
    ):
        return "complex"
    raise ValueError(f"Undefined consortium for {file_name}")


def calc_metrics(
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
):
    return (
        r2_score(
            y_true=y_train_true.flatten(),
            y_pred=y_train_pred.flatten(),
        ),
        r2_score(
            y_true=y_test_true.flatten(),
            y_pred=y_test_pred.flatten(),
        ),
        calc_rmse(
            y_true=y_train_true,
            y_pred=y_train_pred,
        ),
        calc_rmse(
            y_true=y_test_true,
            y_pred=y_test_pred,
        ),
    )


def process_future_outlook(
    file_name,
):
    logger.info(f"Processing future outlook: {file_name}")
    data = np.load(
        DIR_CACHE_CONSORTIA
        / file_name,
    )
    _, rmse_test, _ = get_rmse(
        y_true=data["w_test"],
        y_pred=data["w_test_pred"],
    )
    r2_test = r2_score(
        y_true=data["w_test"].reshape(-1, data["w_test"].shape[2])[:, SEQ_LEN:],
        y_pred=data["w_test_pred"].reshape(-1, data["w_test"].shape[2])[:, SEQ_LEN:],
        multioutput="raw_values",
    )
    
    result = {
        "file_type": "future_outlook",
        "file_name": file_name,
        "w_test.shape": data["w_test"].shape,
        "r2_test": r2_test,
        "rmse_test": rmse_test,
        "example_test": (data["w_test"][0], data["w_test_pred"][0],),
    }
    logger.info(f"Returning result for future outlook: {file_name}")
    return result


def main():
    for _, _, files in os.walk(DIR_CACHE_CONSORTIA):
        break

    file_names = []
    for file_name in files:
        if file_name.find("future_outlook") == -1:
            continue
        file_names.append(file_name)
    logger.info("Beginning pool")
    with Pool() as pool:
        results = pool.map(process_future_outlook, file_names,)

    df = pd.DataFrame.from_records(results)

    logger.info("Saving pickled forecast results")
    with open(
        DIR_RESULTS_CONSORTIA
        / "consortia_sim_forecast.pkl",
        "wb",
    ) as fp:
        pickle.dump(df, fp)

    # df["consortium"] = [("zz294_VAE_saved_sims_" + fn.split("zz294_VAE_saved_sims_")[1].split("_fixed")[0] + "_fixed") for fn in df.file_name]
    df["consortium"] = [define_consortium(file_name=fn) for fn in df.file_name]
    df["input_type"] = [fn.split("future_outlook_")[1].split("_")[0] for fn in df.file_name]
    df["tgt_type"] = [fn.split("_202")[0].split("_")[-1] for fn in df.file_name]
    df["train_size"] = [int(fn.split("_" + NOW_TEXT)[0].split("_")[-1]) for fn in df.file_name]
    df["cross_val"] = [int(fn.split("_" + NOW_TEXT)[0].split("_")[-2]) for fn in df.file_name]
    results = []
    for consortium in df.consortium.unique():
        mask_consortium = df.consortium == consortium
        for train_size in df.loc[mask_consortium, "train_size"].unique():
            mask_ts = mask_consortium & (df.train_size == train_size)
            for input_type in df.loc[mask_ts, "input_type"].unique():
                mask_input_type = mask_ts & (df.input_type == input_type)
                r2 = np.zeros([5, 640])
                rmse = np.zeros([5, 640])
                for i, cross_val in enumerate(df.loc[mask_input_type, "cross_val"].unique()):
                    r2[i, :] = df.loc[mask_input_type & (df.cross_val == cross_val), "r2_test"].iloc[0]
                    rmse[i, :] = df.loc[mask_input_type & (df.cross_val == cross_val), "rmse_test"].iloc[0]
                results.append(
                    {
                        "consortium": consortium,
                        "train_size": train_size,
                        "input_type": input_type,
                        "r2_test": r2.mean(axis=0),
                        "rmse_test": rmse.mean(axis=0),
                        "test_example": df.loc[mask_input_type, "example_test"].iloc[0],
                        "correct_num_cross_val": i == 4,
                    }
                )
    df_out = pd.DataFrame.from_records(results)
    with open(
        DIR_RESULTS_CONSORTIA
        / "consortia_sim_for_figs.pkl",
        "wb",
    ) as fp:
        pickle.dump(df_out, fp)
