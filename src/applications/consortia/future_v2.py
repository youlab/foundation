import json
import logging

import numpy as np

from applications.antibiotics.generate_results import main as generate_results
from applications.consortia.data import get_data
from config import (
    DIR_CACHE_CONSORTIA,
    PATH_DATA_ZZ294_COMPLEX,
    PATH_DATA_ZZ294_SIMPLE,
    MODEL_TYPE,
    SEQ_LEN,
    Z_DIM,
)
from ml.utils.load_models import load_default_model

logger = logging.getLogger(__name__)

# TODO: Use a boolean value to check cache and if it exists, then return early

def main(
    task_id,
    max_depth,
):
    logger.info(f"=== STARTING future_v2.main(task_id={task_id}, max_depth={max_depth}) ===")
    
    # Check if results already exist
    result_file = DIR_CACHE_CONSORTIA / f"consortia_regression_{task_id}_{MODEL_TYPE}_{Z_DIM}.json"
    logger.info(f"Checking for existing results at: {result_file}")
    
    try:
        with open(result_file, "r") as fp:
            json.load(fp)
        logger.info(f"Results file {result_file.name} already exists, returning early.")
        return
    except FileNotFoundError:
        logger.info(f"Results file {result_file.name} not found, proceeding with computation.")
    except Exception as e:
        logger.warning(f"Exception loading {result_file.name}: {e}")
        logger.info("Proceeding with computation despite exception.")

    # Configuration
    INTERP_LEN = SEQ_LEN * 6
    input_type = "raw"
    train_size = 8_000
    
    logger.info(f"Configuration: INTERP_LEN={INTERP_LEN}, input_type={input_type}, train_size={train_size}")

    # Load model
    logger.info("Loading default model...")
    model = load_default_model()
    logger.info("Model loaded successfully")

    # Define parameter combinations
    params = [
        (PATH_DATA_ZZ294_SIMPLE, "segment128", "simple"),
        (PATH_DATA_ZZ294_SIMPLE, "latent", "simple"),
        (PATH_DATA_ZZ294_COMPLEX, "segment128", "complex"),
        (PATH_DATA_ZZ294_COMPLEX, "latent", "complex"),
    ]
    
    logger.info(f"Total parameter combinations: {len(params)}")

    # Select parameters based on task_id
    file_path, tgt_type, name = params[task_id % 4]
    cross_val = task_id // 4

    name_suffix = f"{name}_{tgt_type}_{MODEL_TYPE}_{Z_DIM}"
    
    logger.info(f"=== FUTURE_V2 PARAMETER DEBUG ===")
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Parameter index (task_id % 4): {task_id % 4}")
    logger.info(f"Cross validation index (task_id // 4): {cross_val}")
    logger.info(f"Selected parameters for task_id {task_id}:")
    logger.info(f"  - file_path: {file_path}")
    logger.info(f"  - tgt_type: {tgt_type}")
    logger.info(f"  - name: {name}")
    logger.info(f"  - cross_val: {cross_val}")
    logger.info(f"  - name_suffix: {name_suffix}")
    logger.info(f"  - input_type: {input_type} (hardcoded in future_v2)")
    logger.info(f"  - train_size: {train_size}")
    
    # Calculate what regression files will be created
    train_subsets = [0.1, 0.4, 0.8]
    logger.info(f"Expected regression files to be created:")
    for subset in train_subsets:
        n = int(train_size * subset * 16)
        regr_filename = f"regr_{input_type}_{name_suffix}_{cross_val}_{n}.pkl"
        logger.info(f"  - {regr_filename} (subset={subset}, n={n})")
    logger.info(f"=== END FUTURE_V2 PARAMETER DEBUG ===")
    # Load data
    logger.info("Loading data...")
    (
        raw_data,
        dataset_train,
        dataset_test,
    ) = get_data(
        file_path=file_path,
        interp_len=INTERP_LEN,
        dir_cache=None,
        train_size=train_size,
        use_raw=input_type == "raw",
        tgt_type=tgt_type,
        model=model,
        model_type=MODEL_TYPE,
        z_dim=Z_DIM,
    )
    logger.info("Data loaded successfully")
    logger.info(f"Raw data n_focal: {raw_data.n_focal}")

    # Prepare data for training the regression models
    logger.info("Reshaping and preparing training data...")
    x_train = dataset_train.x.reshape(dataset_train.x.shape[0], -1)
    x_test = dataset_test.x.reshape(dataset_test.x.shape[0], -1)
    tgt_train = dataset_train.tgt.reshape(dataset_train.tgt.shape[0], -1)
    tgt_test = dataset_test.tgt.reshape(dataset_test.tgt.shape[0], -1)
    
    logger.info(f"Data shapes after reshape:")
    logger.info(f"  - x_train: {x_train.shape}")
    logger.info(f"  - x_test: {x_test.shape}")
    logger.info(f"  - tgt_train: {tgt_train.shape}")
    logger.info(f"  - tgt_test: {tgt_test.shape}")

    # Handle NaN values
    logger.info("Handling NaN values...")
    x_train[np.isnan(x_train)] = 0
    x_test[np.isnan(x_test)] = 0
    tgt_train[np.isnan(tgt_train)] = 0
    tgt_test[np.isnan(tgt_test)] = 0
    logger.info("NaN values replaced with 0")

    logger.info(f"Starting generate_results for {name_suffix}")
    # Generate results - this is the main computational step
    try:
        results = generate_results(
            x_raw=x_train,
            tgt=tgt_train,
            z_dim=Z_DIM,
            model_type=MODEL_TYPE,
            classify=False,
            detailed_binary=False,
            epochs_fine_tuned=0,
            lr_fine_tuned=0.1,
            epochs_end2end=0,
            lr_end2end=0.1,
            return_latent_vectors=False,
            stack=raw_data.n_focal,
            cross_val=cross_val,
            k_fold=5,
            prediction_model_cache_dir=DIR_CACHE_CONSORTIA,
            name_suffix=name_suffix,
            train_sizes=[0.1, 0.4, 0.8,],
            max_depth=max_depth,
        )
        logger.info("Results generated successfully")
    except Exception as e:
        logger.error(f"Error in generate_results: {e}")
        raise

    # Save results
    logger.info(f"Saving results to {result_file}")
    try:
        with open(result_file, "w") as fp:
            json.dump(results, fp)
        logger.info("Results saved successfully")
        logger.info(f"=== COMPLETED future_v2.main for task_id={task_id} ===")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise
