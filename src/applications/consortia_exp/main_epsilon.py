import logging
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from applications.config import NOW_TEXT
from applications.consortia_exp.config import get_config
from applications.consortia_exp.main_predictions import get_focus_params
from applications.utils.metrics import (
    calc_rmse,
    get_rmse,
)
from config import (
    DIR_CACHE_CONSORTIA_EXP,
    DIR_RESULTS_CONSORTIA_EXP,
)

logger = logging.getLogger(__name__)


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


def process_regr_pred(
    file_path,
):
    data = np.load(file_path)
    file_name = str(file_path).split("/")[-1]
    _, _, _, input_type, tgt_type, _ = file_name.split("_")
    r2_train, r2_test, rmse_train, rmse_test = calc_metrics(
        y_train_true=data["tgt_train"],
        y_train_pred=data["tgt_train_pred"],
        y_test_true=data["tgt_test"],
        y_test_pred=data["tgt_test_pred"],
    )
    return {
        "file_type": "regr_pred",
        "file_name": file_name,
        "input_type": input_type,
        "tgt_type": tgt_type,
        "x_train.shape": data["x_train"].shape,
        "x_test.shape": data["x_test"].shape,
        "tgt_train.shape": data["tgt_train"].shape,
        "tgt_test.shape": data["tgt_test"].shape,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
    }


def process_future_outlook(
    file_path,
    n_focal,
):
    data = np.load(file_path)
    file_name = str(file_path).split("/")[-1]
    _, _, source_file_rep, input_type, tgt_type, _ = file_name.split("_")
    source_file = source_file_rep.split("-")[0] + "-" + source_file_rep.split("-")[1]
    test_rep = source_file = source_file_rep.split("-")[2]
    _, rmse_train, _ = get_rmse(
        y_true=data["w_train"],
        y_pred=data["w_train_pred"],
    )
    _, rmse_test, _ = get_rmse(
        y_true=data["w_test"],
        y_pred=data["w_test_pred"],
    )

    result = {
        "file_type": "future_outlook",
        "n_focal": n_focal,
        "file_name": file_name,
        "source_file": source_file,
        "rep": test_rep,
        "input_type": input_type,
        "tgt_type": tgt_type,
        "w_train.shape": data["w_train"].shape,
        "w_test.shape": data["w_test"].shape,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "example_train": (data["w_train"][0], data["w_train_pred"][0]),
        "example_test": (data["w_test"][0], data["w_test_pred"][0]),
    }
    return result, data["w_test"], data["w_test_pred"]


def process_param(param):
    source_file, rep_idx, input_type, tgt_type = param
    strains, test_reps = get_config(source_file=source_file)
    test_rep = test_reps[rep_idx]
    name_suffix = f"{source_file}-{test_rep}_{input_type}_{tgt_type}"
    try:
        regr_pred = process_regr_pred(
            file_path=DIR_CACHE_CONSORTIA_EXP
            / f"regr_pred_{name_suffix}_{NOW_TEXT}.npz",
        )

        future_outlook, y_true, y_pred = process_future_outlook(
            file_path=DIR_CACHE_CONSORTIA_EXP
            / f"future_outlook_{name_suffix}_{NOW_TEXT}.npz",
            n_focal=len(strains),
        )
        return {
            "name_suffix": name_suffix,
            "regr_pred": regr_pred,
            "future_outlook": future_outlook,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    except Exception as e:
        logger.warning(f"Exception processing param with {name_suffix}: {e}")
    # FIXME: remove this, only used for troubleshooting before code is done running
    return {}
    

def get_data():
    params = get_focus_params()
    usable_params = []
    for param in params:
        _, _, input_type, tgt_type = param
        if (
            input_type == "raw"
        ) and (
            tgt_type == "latent"
        ):
            continue
        if (
            input_type == "latent"
        ) and (
            tgt_type == "segment128"
        ):
            continue
        usable_params.append(param)
        
    with Pool() as pool:
        results = pool.map(process_param, usable_params)
    return results


def main():
    results = get_data()
    with open(
        DIR_RESULTS_CONSORTIA_EXP
        / "consortia_exp_cache.pkl",
        "wb",
    ) as fp:
        pickle.dump(results, fp)
    rmses = {}
    for result in results:
        # FIXME: remove this, only used for troubleshooting before code is done running
        if result == {}:
            continue
        rmses[result["name_suffix"]] = calc_rmse(
            y_true=result["y_true"],
            y_pred=result["y_pred"],
        )
    df = pd.DataFrame.from_dict(rmses, orient="index").reset_index().rename(columns={"index": "id", 0: "rmse"})
    df["experiment"] = [val[0] for val in df.id.str.split("_")]
    df["input_type"] = [val[1] for val in df.id.str.split("_")]
    df["tgt_type"] = [val[2] for val in df.id.str.split("_")]
    df["test_rep"] = [val[-1] for val in df.experiment]
    df["experiment"] = [val[:7] for val in df.experiment]
    df["experiment"] = [val[0] + "-" + val[1] for val in df.experiment.str.split("-")]

    df.to_csv(
        DIR_RESULTS_CONSORTIA_EXP
        / "consortia_exp_fc_all.csv",
    )
    df.groupby(["experiment", "input_type", "tgt_type",]).agg("mean", numeric_only=True,).reset_index().to_csv(
        DIR_RESULTS_CONSORTIA_EXP
        / "consortia_exp_forecast_rmse_epsilon.csv",
    )
