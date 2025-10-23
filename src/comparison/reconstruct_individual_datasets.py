import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm

from applications.utils.latents import reconstruct
from config import (
    DIR_RESULTS_MODEL_COMPARISON,
    MODEL_TYPE,
    Z_DIM,
)
from data.utils import get_data
from ml.utils.load_models import load_default_model
    


def get_recons(
    train_all,
    test_all,
    model,
    batch_size,
    return_recons=False,
    is_vae=True,
    is_pca=False,
):
    recon_train_all = reconstruct(
        model=model,
        x=train_all,
        batch_size=batch_size,
        is_vae=is_vae,
        is_pca=is_pca,
    )
    recon_test_all = reconstruct(
        model=model,
        x=test_all,
        batch_size=batch_size,
        is_vae=is_vae,
        is_pca=is_pca,
    )

    if return_recons:
        return (
            recon_train_all,
            recon_test_all,
        )

    r2_train_all = r2_score(
        y_true=train_all.flatten(),
        y_pred=recon_train_all.flatten(),
    )
    
    r2_test_all = r2_score(
        y_true=test_all.flatten(),
        y_pred=recon_test_all.flatten(),
    )

    return (
        r2_train_all,
        r2_test_all,
    )


def main():
    data, idx = get_data()
    y = data["y"]
    model = load_default_model()
    results = []
    for file_name in tqdm(idx["experimental"].keys()):
        i1 = idx["experimental"][file_name]["y_all_i"]
        i2 = i1 + idx["experimental"][file_name]["y.shape"][0]

        trains, tests = [], []
        for val in np.arange(i1, i2,):
            if val in data["train_idx"]:
                trains.append(y[val, :].reshape(1, -1))
            elif val in data["test_idx"]:
                tests.append(y[val, :].reshape(1, -1))
            else:
                raise ValueError
        if (
            len(trains) == 0
        ) or (
            len(tests) == 0
        ):
            continue
        x_train = np.concatenate(trains)
        x_test = np.concatenate(tests)
        
        (
            r2_train_all,
            r2_test_all,
        ) = get_recons(
            train_all=x_train,
            test_all=x_test,
            model=model,
            batch_size=4_096,
            return_recons=False,
            is_vae=True,
        )

        results.append(
            {
                "file_name": file_name,
                "r2_train_all": r2_train_all,
                "r2_test_all": r2_test_all,
            }
        )

    print("Saving first csv")
    pd.DataFrame.from_records(
        data=results,
    ).to_csv(
        DIR_RESULTS_MODEL_COMPARISON
        / f"recon_by_dataset_{MODEL_TYPE}_{Z_DIM}.csv",
    )
