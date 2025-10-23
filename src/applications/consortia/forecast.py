import logging

from tqdm import tqdm

from applications.consortia.data import get_data
from applications.consortia.future_outlook import main as main_future_outlook
from ml.utils.load_models import load_default_model
from config import (
    DIR_CACHE_CONSORTIA,
    PATH_DATA_ZZ294_COMPLEX,
    PATH_DATA_ZZ294_SIMPLE,
    MODEL_TYPE,
    SEQ_LEN,
    Z_DIM,
)

logger = logging.getLogger(__name__)

def main(
    task_id,
):
    INTERP_LEN = SEQ_LEN * 6

    model = load_default_model()

    params = [
        (
            PATH_DATA_ZZ294_SIMPLE,
            "segment128",
            "simple",
        ),
        (
            PATH_DATA_ZZ294_SIMPLE,
            "latent",
            "simple",
        ),
        (
            PATH_DATA_ZZ294_COMPLEX,
            "segment128",
            "complex",
        ),
        (
            PATH_DATA_ZZ294_COMPLEX,
            "latent",
            "complex",
        ),
    ]

    file_path, tgt_type, name = params[task_id % 4]
    input_type = "latent" if (tgt_type == "latent") else "raw"
    cross_val = task_id // 4

    name_suffix = f"{name}_{tgt_type}_{MODEL_TYPE}_{Z_DIM}"
    
    logger.info(f"=== FORECAST PARAMETER DEBUG ===")
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Parameter index (task_id % 4): {task_id % 4}")
    logger.info(f"Cross validation index (task_id // 4): {cross_val}")
    logger.info(f"Selected parameters:")
    logger.info(f"  - file_path: {file_path}")
    logger.info(f"  - tgt_type: {tgt_type}")
    logger.info(f"  - name: {name}")
    logger.info(f"  - input_type: {input_type}")
    logger.info(f"  - name_suffix: {name_suffix}")
    logger.info(f"Forecasting for {name_suffix}")
    logger.info(f"=== END FORECAST PARAMETER DEBUG ===")
    
    for train_size in [
        8_000,
    ]:
        raw_data = get_data(
            file_path=file_path,
            interp_len=INTERP_LEN,
            dir_cache=None,
            train_size=train_size,
            use_raw=True,
            tgt_type=tgt_type,
            model=model,
            model_type=MODEL_TYPE,
            z_dim=Z_DIM,
            return_raw_data_only=True,
        )
        for subset in tqdm(
            [
                0.1,
                0.4,
                0.8,
            ],
            f"Forecasting for train_size={train_size}",
        ):
            n = int(train_size * subset * 16)
            forecast_name_suffix = f"{input_type}_{name_suffix}_{cross_val}_{n}"
            
            logger.info(f"=== FORECAST CALL DEBUG ===")
            logger.info(f"Subset: {subset}")
            logger.info(f"Calculated n: {n} (train_size={train_size} * subset={subset} * 16)")
            logger.info(f"Expected regression file: regr_{forecast_name_suffix}.pkl")
            logger.info(f"Full path will be: {DIR_CACHE_CONSORTIA}/regr_{forecast_name_suffix}.pkl")
            logger.info(f"Parameters passed to main_future_outlook:")
            logger.info(f"  - use_raw: {input_type == 'raw'}")
            logger.info(f"  - tgt_type: {tgt_type}")
            logger.info(f"  - name: {name}")
            logger.info(f"  - name_suffix: {forecast_name_suffix}")
            logger.info(f"=== END FORECAST CALL DEBUG ===")
            
            main_future_outlook(
                raw_data=raw_data,
                use_raw=input_type == "raw",
                tgt_type=tgt_type,
                dir_cache=DIR_CACHE_CONSORTIA,
                name=name,
                name_suffix=forecast_name_suffix,
                model=model,
                model_type=MODEL_TYPE,
                z_dim=Z_DIM,
            )
