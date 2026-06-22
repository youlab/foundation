from config import (
    DIR_DATA_KARLSSON,
    SEQ_LEN,
    Z_DIM,
)

import os

if __name__ == "__main__":

    RUN_KARLSSON_DARK = True

    if RUN_KARLSSON_DARK:

        from applications.karlsson.forecast import main
        random_seed = 501
        results = main(
            data_dir=DIR_DATA_KARLSSON,
            input_type='latent', 
            target_type='latent',
            split_level='per_replicate',
            train_size=0.8, 
            window_size=SEQ_LEN,
            stride=32,
            max_depth=15,
            random_seed=random_seed,
        )