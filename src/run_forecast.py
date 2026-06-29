from config import (
    DIR_DATA_KARLSSON,
    SEQ_LEN,
    Z_DIM,
)

import os

if __name__ == "__main__":

    RUN_KARLSSON_DARK = False
    RUN_KARLSSON_DARK_SAMPLE_SIZE_ANALYSIS = True

    if RUN_KARLSSON_DARK:

        from applications.karlsson.forecast import main
        print("Running RUN_KARLSSON_DARK")
        results = main(
            data_dir=DIR_DATA_KARLSSON,
            input_type='pca', 
            target_type='raw',
            use_test_indices_at='full_split_indices_seed501.json',
            split_level='per_replicate',
            train_size=0.8, 
            train_downsample_to_n=10,
            window_size=SEQ_LEN,
            stride=128,
            max_depth=15,
            random_seed=501,
            downsampling_seed=627,
        )
 
    if RUN_KARLSSON_DARK_SAMPLE_SIZE_ANALYSIS:

        from applications.karlsson.forecast_sample_size import main
        print("Running RUN_KARLSSON_DARK_SAMPLE_SIZE_ANALYSIS")
        results = main(
            n_repeats=5,
            random_seed=501,
            base_downsampling_seed=627,
            stride=128,
            max_depth=15,
            plot_only=True, # if True, assumes results from running with plot_only=False are present
            plot_start_idx=1, # index of train size to start plotting
        )