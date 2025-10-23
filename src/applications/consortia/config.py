from itertools import product

import numpy as np

from config import (
    PATH_DATA_ZZ294_SIMPLE,
    PATH_DATA_ZZ294_COMPLEX,
)


def get_params(task_id=None):
    file_path_indices = list(np.arange(150))
    train_sizes = [
        125,
        500,
        2000,
        8000,
    ]

    tgt_types = [
        "segment16",
        "segment32",
        "segment128",
    ]

    use_raws = [
        False,
        True,
    ]
    
    params = list(
        product(
            file_path_indices,
            train_sizes,
            tgt_types,
            use_raws,
        )
    )

    if task_id is None:
        return params
    return params[task_id]


def get_consortia(task_id=None):
    from itertools import product

    file_paths = [
        # moderate, IDE = 3
        PATH_DATA_ZZ294_SIMPLE,
        # chaotic, IDE = 6
        PATH_DATA_ZZ294_COMPLEX,
    ]
    train_sizes = [200, 8_000,]
    input_types = ["raw", "latent",]
    tgt_types = ["segment128", "latent", "segment16", "sliders16", "sliders32", "sliders64",]
    params = list(product(file_paths, train_sizes, input_types, tgt_types,))
    if task_id is None:
        return params
    return params[task_id]


def get_consortia_for_plotting():
    return [
        # moderate, IDE = 3
        # "zz294_VAE_saved_sims_bgLV_I7_bgLV_B15_T5_fixed",
        "simple",
        # chaotic, IDE = 6
        # "zz294_VAE_saved_sims_dgLV_I5_dgLV_B92_T8_fixed",
        "complex",
    ]


def get_consortia_for_intrinsic_dims():
    return [
        # moderate, IDE = 3
        "zz294_VAE_saved_sims_bgLV_I7_bgLV_B15_T5_fixed",
        # "simple",
        # chaotic, IDE = 6
        "zz294_VAE_saved_sims_dgLV_I5_dgLV_B92_T8_fixed",
        # "complex",
    ]
