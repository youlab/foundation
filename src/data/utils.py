import json

import numpy as np

from config import DIR_DATA_PROCESSED


def get_data(
    category="all",
    return_split=False,
):
    if category not in {"all", "simulation", "experimental"}:
        raise ValueError(f"Category should be 'all', 'simulation', or 'experimental', but it is {category}")
    
    data = np.load(
        DIR_DATA_PROCESSED
        / f"128_2024-08-16_{category}.npz",
    )
    if return_split:
        return (
            data["y"][data["train_idx"]],
            data["y"][data["test_idx"]],
        )

    with open(
        DIR_DATA_PROCESSED
        / f"128_2024-08-16_{category}_idx_key.json",
        "r",
    ) as fp:
        idx = json.load(fp)
    return data, idx
