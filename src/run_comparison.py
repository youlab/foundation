def get_task_id():
    import os
    return int(os.environ["SLURM_ARRAY_TASK_ID"])


if __name__ == "__main__":
    RUN_RECON_FULL = False
    RUN_RECON_SUMMARY = False
    RUN_RECON_INDIVIDUAL_DATASETS = False

    RUN_ANTIBIOTICS_FULL = False  # good. use array length 41 for slurm
    RUN_ANTIBIOTICS_SUMMARY = True

    RUN_PLOT_SUMMARIES = False

    RUN_ANTIBIOTICS_FINAL_SUMMARY = False

    if RUN_ANTIBIOTICS_FINAL_SUMMARY:
        from comparison.final_antibiotics_summary import main
        print("Running RUN_ANTIBIOTICS_FINAL_SUMMARY")
        main()

    if RUN_RECON_FULL:
        from comparison.reconstruction import main
        from ml.utils.load_models import get_best_models

        task_id = get_task_id()
        models = get_best_models()
        main(
            model_type=models[task_id][0],
            z_dim=models[task_id][1],
            batch_size=1_024,
            check_cache=True,
        )

    if RUN_RECON_SUMMARY:
        print("Running reconstruction summary")
        from comparison.summarize_reconstruction import main
        main()

    if RUN_ANTIBIOTICS_FULL:
        from applications.antibiotics.main import main as run_antibiotics
        from ml.utils.load_models import get_best_models

        task_id = get_task_id()
        models = get_best_models()
        model_type = models[task_id][0]
        z_dim = models[task_id][1]

        for cross_val in range(5):
            run_antibiotics(
                model_type=model_type,
                z_dim=z_dim,
                cross_val=cross_val,
            )
        from applications.antibiotics.kyeri.write_antibiotic_summary import main as main_kyeri
        from applications.antibiotics.carolyn.write_antibiotic_summary import main as main_carolyn
        main_kyeri(
            model_type=model_type,
            z_dim=z_dim,
        )

        main_carolyn(
            model_type=model_type,
            z_dim=z_dim,
        )


    if RUN_ANTIBIOTICS_SUMMARY:
        from comparison.summarize_antibiotics import main
        print("Running RUN_ANTIBIOTICS_SUMMARY")
        main()
    
    if RUN_RECON_INDIVIDUAL_DATASETS:
        from comparison.reconstruct_individual_datasets import main
        print("Running RUN_RECON_INDIVIDUAL_DATASETS")
        main()
