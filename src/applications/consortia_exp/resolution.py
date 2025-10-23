import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from applications.consortia_exp.config import get_config
from applications.utils.data import interpolate_y
from applications.utils.latents import get_latents
from ml.utils.load_models import load_default_model
from config import (
    DIR_RESULTS_CONSORTIA_EXP,
    PATH_FUJITA_DATA,
)


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


def get_params():
    t_cols = [str(val) for val in np.arange(1, 111)]
    days_outs = [1, 2, 4, 8,]
    n_input = 16
    l_total = len(t_cols)
    n_windows = l_total - n_input - days_outs[-1]
    reduxes = [1, 2, 4, 8,]
    return t_cols, days_outs, n_input, l_total, n_windows, reduxes


def get_data():
    source_files = [
        "soil-a",
        "soil-b",
        "soil-c",
        "water-a",
        "water-b",
        "water-c",
    ]
    data = pd.read_csv(PATH_FUJITA_DATA)
    return data, source_files


def get_variables(
    source_files,
    data,
    t_cols,
    days_outs,
    n_windows,
    n_input,
):
    variables = {}

    for source_file in source_files:
        strains, test_reps = get_config(source_file=source_file)
        mask_strains = np.array([strain in strains for strain in data.strain])
        mask_sf = mask_strains & (data.file == source_file)
        variables[source_file] = {}
        for rep in test_reps:
            mask_rep = mask_sf & (data.rep == rep)
            variables[source_file][rep] = {}
            for days_out in days_outs:
                variables[source_file][rep][days_out] = {
                    "x": [],
                    "tgt": [],
                }
                for window in range(n_windows):
                    variables[source_file][rep][days_out]["x"].append(data.loc[mask_rep, t_cols[window:window+n_input]].to_numpy())
                    variables[source_file][rep][days_out]["tgt"].append(data.loc[mask_rep, t_cols[window + n_input + days_out]].to_numpy())

    return variables


def get_x_and_tgt(
    variables,
    source_file,
    test_data_rep,
    days_out,
    n_input,
):
    _, test_reps = get_config(source_file=source_file)

    x_train, x_test = None, None
    tgt_train, tgt_test = None, None,
    for rep in test_reps:
        if rep == test_data_rep:
            x_test = np.array(variables[source_file][rep][days_out]["x"])
            n_focal = x_test.shape[1]
            tgt_test = np.array(variables[source_file][rep][days_out]["tgt"]).reshape(-1, n_focal, 1)
        elif x_train is None:
            x_train = np.array(variables[source_file][rep][days_out]["x"])
            n_focal = x_train.shape[1]
            tgt_train = np.array(variables[source_file][rep][days_out]["tgt"]).reshape(-1, n_focal, 1)
        else:
            x_train = np.concatenate(
                (
                    x_train,
                    np.array(variables[source_file][rep][days_out]["x"]),
                )
            )
            tgt_train = np.concatenate(
                (
                    tgt_train,
                    np.array(variables[source_file][rep][days_out]["tgt"]).reshape(-1, n_focal, 1),
                )
            )

    n_train, n_focal, _ = x_train.shape
    n_test = x_test.shape[0]

    x_train = x_train / x_train.sum(axis=1).reshape(n_train, 1, n_input)
    x_test = x_test / x_test.sum(axis=1).reshape(n_test, 1, n_input)
    tgt_train = tgt_train / tgt_train.sum(axis=1).reshape(n_train, 1, 1)
    tgt_test = tgt_test / tgt_test.sum(axis=1).reshape(n_test, 1, 1)

    x_train[np.isnan(x_train)] = 0
    x_test[np.isnan(x_test)] = 0
    tgt_train[np.isnan(tgt_train)] = 0
    tgt_test[np.isnan(tgt_test)] = 0

    tgt_train = (tgt_train > 0.1) * 1
    tgt_test = (tgt_test > 0.1) * 1

    return x_train, x_test, tgt_train, tgt_test, n_train, n_test, n_focal


def get_raw_and_lat(
    x_train,
    x_test,
    redux,
    n_train,
    n_test,
    n_focal,
    n_input,
    model,
):
    x_train_raw = x_train[:, :, 0::redux]
    x_test_raw = x_test[:, :, 0::redux]

    x_train_lat = get_latents(
        z=interpolate_y(y=x_train_raw.reshape(n_train * n_focal, n_input // redux)),
        model=model,
        batch_size=4_096,
    ).reshape(n_train, n_focal, 9)
    x_test_lat = get_latents(
        z=interpolate_y(y=x_test_raw.reshape(n_test * n_focal, n_input // redux)),
        model=model,
        batch_size=4_096,
    ).reshape(n_test, n_focal, 9)
    return x_train_raw, x_test_raw, x_train_lat, x_test_lat

def main(
    task_id,
):
    print("Beginning resolution")
    model = load_default_model()
    print("Loading data")
    data, source_files = get_data()
    source_file = source_files[task_id]
    t_cols, days_outs, n_input, l_total, n_windows, reduxes = get_params()
    print("Getting variables")
    variables = get_variables(
        source_files=source_files,
        data=data,
        t_cols=t_cols,
        days_outs=days_outs,
        n_windows=n_windows,
        n_input=n_input,
    )

    _, test_reps = get_config(source_file=source_file)
    results = {}
    print("Generating results")
    for test_data_rep in tqdm(test_reps, "Working through test reps"):
        results[test_data_rep] = {}
        for days_out in days_outs:
            results[test_data_rep][days_out] = {}
            x_train, x_test, tgt_train, tgt_test, n_train, n_test, n_focal = get_x_and_tgt(
                variables=variables,
                n_input=n_input,
                source_file=source_file,
                test_data_rep=test_data_rep,
                days_out=days_out,
            )
            for redux in reduxes:
                x_train_raw, x_test_raw, x_train_lat, x_test_lat = get_raw_and_lat(
                    x_train=x_train,
                    x_test=x_test,
                    redux=redux,
                    n_train=n_train,
                    n_test=n_test,
                    n_focal=n_focal,
                    n_input=n_input,
                    model=model,
                )
                tgt_test_pred_raw = train_classifier(
                    x_train=x_train_raw.reshape(n_train, -1),
                    x_test=x_test_raw.reshape(n_test, -1),
                    tgt_train=tgt_train.reshape(n_train, -1),
                )
                tgt_test_pred_lat = train_classifier(
                    x_train=x_train_lat.reshape(n_train, -1),
                    x_test=x_test_lat.reshape(n_test, -1),
                    tgt_train=tgt_train.reshape(n_train, -1),
                )
                results[test_data_rep][days_out][redux] = {
                    "tgt_test": tgt_test,
                    "tgt_test_pred_raw": tgt_test_pred_raw,
                    "tgt_test_pred_lat": tgt_test_pred_lat,
                    "acc_raw": accuracy_score(
                        y_true=tgt_test.flatten(),
                        y_pred=tgt_test_pred_raw.flatten(),
                    ),
                    "acc_lat": accuracy_score(
                        y_true=tgt_test.flatten(),
                        y_pred=tgt_test_pred_lat.flatten(),
                    ),
                }
    with open(
        DIR_RESULTS_CONSORTIA_EXP
        / f"fut_abun_{source_file}_v3.pkl",
        "wb",
    ) as fp:
        pickle.dump(
            obj=results,
            file=fp,
        )
