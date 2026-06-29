import warnings
import logging
import os
import traceback

from config import (
    MODEL_TYPE,
    Z_DIM,
)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('run_applications.log', mode='a')  # File output
    ]
)
logger = logging.getLogger(__name__)


def get_task_id(debug=False):
    if debug:
        logger.info("Using debug mode, task_id=0")
        return 0
    
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_str is None:
        logger.error("SLURM_ARRAY_TASK_ID not found in environment")
        raise RuntimeError("SLURM_ARRAY_TASK_ID environment variable not set")
    
    task_id = int(task_id_str)
    logger.info(f"SLURM task ID: {task_id}")
    return task_id


if __name__ == "__main__":
    logger.info("=== STARTING RUN_APPLICATIONS.PY ===")
    logger.info(f"Configuration: MODEL_TYPE={MODEL_TYPE}, Z_DIM={Z_DIM}")
    
    try:
        # Configuration flags
        RUN_CONSORTIA_SIM_V2 = False  # Run with array 0-19
        RUN_CONSORTIA_SIM_FORECAST_V2 = False  # Run with array 0-19 

        RUN_ANTIBIOTIC_MAIN = True
        RUN_ANTIBIOTIC_SUMMARIES = True

        RUN_CONSORTIA_EXP = False  # Run with array 0-479
        # TODO: Remove the cache files and retry it again.

        RUN_CONSORTIA_SIM_FOCUSED_SUMMARY = False # no array needed
        RUN_CONSORTIA_EXP_FOCUSED_SUMMARY = False # no array needed
        RUN_CONSORTIA_EXP_FINAL_ABUNDANCE_CLASSIFICATION = False # no array needed # good

        RUN_SUPER_RESOLUTION = False
        
        # Log enabled applications
        enabled_apps = []
        if RUN_CONSORTIA_SIM_V2: enabled_apps.append("RUN_CONSORTIA_SIM_V2")
        if RUN_CONSORTIA_SIM_FORECAST_V2: enabled_apps.append("RUN_CONSORTIA_SIM_FORECAST_V2")
        if RUN_ANTIBIOTIC_MAIN: enabled_apps.append("RUN_ANTIBIOTIC_MAIN")
        if RUN_ANTIBIOTIC_SUMMARIES: enabled_apps.append("RUN_ANTIBIOTIC_SUMMARIES")
        logger.info(f"Enabled applications: {enabled_apps}")

        if RUN_CONSORTIA_SIM_V2:
            logger.info("=== STARTING RUN_CONSORTIA_SIM_V2 ===")
            
            try:
                # Enable memory tracking
                logger.info("Starting memory tracking")
                import tracemalloc
                tracemalloc.start()
                
                # Get task ID
                logger.info("Getting task ID from SLURM environment")
                task_id = get_task_id(debug=False)
                logger.info(f"Task ID retrieved: {task_id}")
                
                # Import the main function
                logger.info("Importing applications.consortia.future_v2.main")
                from applications.consortia.future_v2 import main
                logger.info("Successfully imported future_v2.main")
                
                # Call the main function
                logger.info(f"Calling main(task_id={task_id}, max_depth=15)")
                main(
                    task_id=task_id,
                    max_depth=3, # change it back to 15 after running.
                )
                logger.info("RUN_CONSORTIA_SIM_V2 completed successfully")
                
            except Exception as e:
                logger.error(f"ERROR in RUN_CONSORTIA_SIM_V2: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise
            finally:
                if tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    logger.info(f"Memory usage - Current: {current / 1024**2:.1f}MB, Peak: {peak / 1024**2:.1f}MB")
                    tracemalloc.stop()
                logger.info("=== FINISHED RUN_CONSORTIA_SIM_V2 ===")
                
            print(f"RUN_CONSORTIA_SIM_V2 completed - check run_applications.log for details")

        if RUN_CONSORTIA_SIM_FORECAST_V2:
            logger.info("=== STARTING RUN_CONSORTIA_SIM_FORECAST_V2 ===")
            
            import tracemalloc
            tracemalloc.start()

            from applications.consortia.forecast import main
            task_id = get_task_id(debug=False)
            
            logger.info(f"=== FORECAST vs FUTURE_V2 FILE NAMING DEBUG ===")
            logger.info(f"This debug section explains the file naming mismatch:")
            logger.info(f"")
            logger.info(f"1. FUTURE_V2 (RUN_CONSORTIA_SIM_V2) saves files with pattern:")
            logger.info(f"   regr_{{input_type}}_{{name_suffix}}_{{cross_val}}_{{train_size}}.pkl")
            logger.info(f"   where input_type is always 'raw' (hardcoded)")
            logger.info(f"")
            logger.info(f"2. FORECAST (RUN_CONSORTIA_SIM_FORECAST_V2) looks for files with pattern:")
            logger.info(f"   regr_{{forecast_name_suffix}}.pkl")
            logger.info(f"   where forecast_name_suffix = {{input_type}}_{{name_suffix}}_{{cross_val}}_{{n}}")
            logger.info(f"   and input_type can be 'raw' or 'latent' depending on tgt_type")
            logger.info(f"")
            logger.info(f"3. For task_id {task_id}:")
            
            # Calculate the parameters the same way forecast.py does
            from config import PATH_DATA_ZZ294_COMPLEX, PATH_DATA_ZZ294_SIMPLE, MODEL_TYPE, Z_DIM
            params = [
                (PATH_DATA_ZZ294_SIMPLE, "segment128", "simple"),
                (PATH_DATA_ZZ294_SIMPLE, "latent", "simple"),
                (PATH_DATA_ZZ294_COMPLEX, "segment128", "complex"),
                (PATH_DATA_ZZ294_COMPLEX, "latent", "complex"),
            ]
            file_path, tgt_type, name = params[task_id % 4]
            input_type = "latent" if (tgt_type == "latent") else "raw"
            cross_val = task_id // 4
            name_suffix = f"{name}_{tgt_type}_{MODEL_TYPE}_{Z_DIM}"
            
            logger.info(f"   - Parameter combination: {file_path.name}, {tgt_type}, {name}")
            logger.info(f"   - Forecast input_type: {input_type} (derived from tgt_type)")
            logger.info(f"   - Future_v2 input_type: raw (hardcoded)")
            logger.info(f"   - Cross validation: {cross_val}")
            logger.info(f"")
            logger.info(f"4. The mismatch occurs when tgt_type='latent' because:")
            logger.info(f"   - Future_v2 saves: regr_raw_{{name_suffix}}_{{cross_val}}_{{n}}.pkl")
            logger.info(f"   - Forecast expects: regr_latent_{{name_suffix}}_{{cross_val}}_{{n}}.pkl")
            logger.info(f"=== END FILE NAMING DEBUG ===")
            logger.info(f"")
            
            main(task_id=task_id)

        if RUN_ANTIBIOTIC_SUMMARIES:
            logger.info("Starting RUN_ANTIBIOTIC_SUMMARIES")
            from applications.antibiotics.kyeri.write_antibiotic_summary import main as main_kyeri
            from applications.antibiotics.carolyn.write_antibiotic_summary import main as main_carolyn
            print("Running RUN_ANTIBIOTIC_SUMMARIES")
            main_kyeri(
                model_type=MODEL_TYPE,
                z_dim=Z_DIM,
            )
            main_carolyn(
                model_type=MODEL_TYPE,
                z_dim=Z_DIM,
            )


        if RUN_CONSORTIA_EXP:
            logger.info("Starting RUN_CONSORTIA_EXP")
            from applications.consortia_exp.main_predictions import main

            task_id = get_task_id()
            print("Running consortia experimental")
            main(task_id=task_id)

        if RUN_ANTIBIOTIC_MAIN:
            logger.info("Starting RUN_ANTIBIOTIC_MAIN")
            from applications.antibiotics.main import main as run_antibiotics
            for cross_val in range(5):
                run_antibiotics(
                    model_type=MODEL_TYPE,
                    z_dim=Z_DIM,
                    cross_val=cross_val,
                )

        if RUN_CONSORTIA_SIM_FOCUSED_SUMMARY:
            logger.info("Starting RUN_CONSORTIA_SIM_FOCUSED_SUMMARY")
            from applications.consortia.main_epsilon import main
            main()

        if RUN_CONSORTIA_EXP_FOCUSED_SUMMARY:
            logger.info("Starting RUN_CONSORTIA_EXP_FOCUSED_SUMMARY")
            from applications.consortia_exp.main_epsilon import main
            print("Running RUN_CONSORTIA_EXP_FOCUSED_SUMMARY")
            main()
    
        if RUN_CONSORTIA_EXP_FINAL_ABUNDANCE_CLASSIFICATION:
            logger.info("Starting RUN_CONSORTIA_EXP_FINAL_ABUNDANCE_CLASSIFICATION")
            from applications.consortia_exp.final_abundance_classification import main
            print("Running FINAL_ABUNDANCE_CLASSIFICATION")
            task_id = get_task_id()
            # thresholds = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,]
            import numpy as np
            # thresholds = np.arange(100).astype(int)
            # for points_back in [1, 10, 20, 40, 80, 160,]:
            main(
                threshold=10,
                points_back=1,
                # threshold=int(thresholds[task_id]),
                # points_back=points_back,
                )


        if RUN_SUPER_RESOLUTION:
            logger.info("Starting RUN_SUPER_RESOLUTION")
            print("Running RUN_SUPER_RESOLUTION")
            from applications.super_resolution.main import main
            main(
                task_id=get_task_id(),
            )
        
        logger.info("=== ALL APPLICATIONS COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"FATAL ERROR in main execution: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        print(f"FATAL ERROR: {str(e)} - Check run_applications.log for details")
        raise
    finally:
        logger.info("=== RUN_APPLICATIONS.PY FINISHED ===")
