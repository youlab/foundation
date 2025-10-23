import pickle

import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm

from applications.config import NOW_TEXT
from applications.consortia.config import get_consortia
from applications.utils.metrics import (
    calc_rmse,
    get_rmse,
)
from config import (
    DIR_CACHE_CONSORTIA,
    SEQ_LEN,
)


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
    consortia, file_name = str(file_path).split("/")[-2:]
    _, _, train_size, input_type, tgt_type, _ = file_name.split("_")
    r2_train, r2_test, rmse_train, rmse_test = calc_metrics(
        y_train_true=data["tgt_train"],
        y_train_pred=data["tgt_train_pred"],
        y_test_true=data["tgt_test"],
        y_test_pred=data["tgt_test_pred"],
    )
    return {
        "file_type": "regr_pred",
        "consortia": consortia,
        "file_name": file_name,
        "train_size": train_size,
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
):
    data = np.load(file_path)
    consortia, file_name = str(file_path).split("/")[-2:]
    _, _, train_size, input_type, tgt_type, _ = file_name.split("_")
    _, rmse_train, _ = get_rmse(
        y_true=data["w_train"],
        y_pred=data["w_train_pred"],
    )
    _, rmse_test, _ = get_rmse(
        y_true=data["w_test"],
        y_pred=data["w_test_pred"],
    )
    r2_train = r2_score(
        y_true=data["w_train"].reshape(-1, data["w_train"].shape[2])[:, SEQ_LEN:],
        y_pred=data["w_train_pred"].reshape(-1, data["w_train"].shape[2])[:, SEQ_LEN:],
        multioutput="raw_values",
    )
    
    r2_test = r2_score(
        y_true=data["w_test"].reshape(-1, data["w_test"].shape[2])[:, SEQ_LEN:],
        y_pred=data["w_test_pred"].reshape(-1, data["w_test"].shape[2])[:, SEQ_LEN:],
        multioutput="raw_values",
    )
    n_focal = int(consortia.split("_T")[1].split("_")[0])

    result = {
        "file_type": "future_outlook",
        "consortia": consortia,
        "n_focal": n_focal,
        "file_name": file_name,
        "train_size": train_size,
        "input_type": input_type,
        "tgt_type": tgt_type,
        "w_train.shape": data["w_train"].shape,
        "w_test.shape": data["w_test"].shape,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "example_train": (data["w_train"][0], data["w_train_pred"][0]),
        "example_test": (data["w_test"][0], data["w_test_pred"][0]),
    }
    return result


def parse_file_path(
        file_path,
):
    consortia = (
        str(file_path)
        .replace("/", "_")
        .replace(".txt", "")
        .replace("_hpc_group_youlab_", "")
    )
    return consortia


def main(return_data=False):
    params = get_consortia()
    regr_pred = []
    future_outlook = []

    for i, (file_path, train_size, input_type, tgt_type,) in enumerate(tqdm(params)):
        consortium = parse_file_path(file_path=file_path)

        file_path_regr_pred = DIR_CACHE_CONSORTIA / f"{consortium}" / f"regr_pred_{train_size}_{input_type}_{tgt_type}_{NOW_TEXT}.npz"
        file_path_future_outlook = DIR_CACHE_CONSORTIA / f"{consortium}" / f"future_outlook_{train_size}_{input_type}_{tgt_type}_{NOW_TEXT}.npz"
        try:
            regr_pred.append(
                process_regr_pred(
                    file_path=file_path_regr_pred,
                )
            )
            future_outlook.append(
                process_future_outlook(
                    file_path=file_path_future_outlook,
                )
            )
        except Exception as e:
            print(f"{i} Exception with {consortium}: {e}")
            continue

    data = {
        "regr_pred": regr_pred,
        "future_outlook": future_outlook,
    }
    if return_data:
        return data

    with open(
        DIR_CACHE_CONSORTIA
        / "data_delta.pkl",
        "wb",
    ) as fp:
        pickle.dump(
            obj=data,
            file=fp,
        )
