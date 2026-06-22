import json

import numpy as np

from config import DIR_DATA_PROCESSED


def get_data(
    run_dir,
    category="all",
    return_split=False,
):
    if category not in {"all", "simulation", "experimental"}:
        raise ValueError(f"Category should be 'all', 'simulation', or 'experimental', but it is {category}")
    
    data_dir = DIR_DATA_PROCESSED / run_dir
    if str(run_dir) == "old_split":
        data = np.load(
            data_dir
            / f"128_2024-08-16_{category}.npz",
        )
        with open(
            data_dir
            / f"128_2024-08-16_{category}_idx_key.json",
            "r",
        ) as fp:
            idx = json.load(fp) 
    
    else:
        data = np.load(
            data_dir
            / f"128_{category}_y.npz",
        )
        with open(
            data_dir
            / f"128_{category}_idx_key.json",
            "r",
        ) as fp:
            idx = json.load(fp)
    
    if return_split:
        return (
            data["y"][data["train_idx"]],
            data["y"][data["test_idx"]],
        )

    return data, idx
