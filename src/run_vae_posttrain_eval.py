if __name__ == "__main__":

    RUN_PLOT_LOSS_CURVES = True
    RUN_PLOT_LATENT_DISTRIBUTION = False
    RUN_PLOT_R2 = False
    RUN_PLOT_PERC_SIM_SENSITIVITY = True

    if RUN_PLOT_LOSS_CURVES:
        print("Running RUN_PLOT_LOSS_CURVES")
        from ml.utils.plot_loss_curves import main
        main()
        
    if RUN_PLOT_LATENT_DISTRIBUTION:
        print("Running RUN_PLOT_LATENT_DISTRIBUTION_NEW_MODELS")
        from pathlib import Path
        from ml.latent_distribution import main
        from ml.config import DATA_RUN_DIR, MODEL_RUN_DIR
        from config import DIR_DATA_PROCESSED, DIR_RESULTS_MODEL_NEW_SPLITS

        model_dir = Path(DIR_RESULTS_MODEL_NEW_SPLITS) / MODEL_RUN_DIR
    
        main(
            data_run_dir=DATA_RUN_DIR,
            model_dir=model_dir,
        )

    if RUN_PLOT_R2:
        print("Running RUN_PLOT_R2")
        from pathlib import Path
        import matplotlib.pyplot as plt
        from figs.model.model_reconstruction_accuracy import plot_model_reconstruction_accuracy
        from ml.config import DATA_RUN_DIR, MODEL_RUN_DIR
        from config import DIR_DATA_PROCESSED, DIR_RESULTS_MODEL_NEW_SPLITS

        model_dir = Path(DIR_RESULTS_MODEL_NEW_SPLITS) / MODEL_RUN_DIR

        fig, ax = plt.subplots(figsize=(6, 3))
        plot_model_reconstruction_accuracy(
            ax=ax,
            data_run_dir=DATA_RUN_DIR,
            model_dir=model_dir,
            fs_ticks=20,
        )
    
    if RUN_PLOT_PERC_SIM_SENSITIVITY:
        print("Running RUN_PLOT_PERC_SIM_SENSITIVITY")
        from ml.perc_sim_sensitivity import main
        main()
