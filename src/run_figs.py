import argparse
import os
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one manuscript figure per invocation.")
    parser.add_argument("--figure", required=True, choices=['presentation', '1', '2', '3', '4', '5', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12'])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Foundation checkout containing the input data and results")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/figs/manuscript",
                        help="Directory for regenerated figures")
    parser.add_argument("--processed-dir", type=Path, help="Directory containing compiled figure_inputs")
    parser.add_argument("--forecast-dir", type=Path, help="Directory containing consortia forecast caches")
    args = parser.parse_args()
    os.environ["FOUNDATION_HOME"] = str(args.root.resolve())
    import config
    if args.processed_dir:
        config.DIR_DATA_PROCESSED = args.processed_dir.resolve()
    if args.forecast_dir:
        config.DIR_CACHE_CONSORTIA = args.forecast_dir.resolve()
        config.DIR_CACHE_CONSORTIA_EXP = args.forecast_dir.resolve()
    config.DIR_FIGS_MANUSCRIPT = args.output_dir.resolve()
    config.DIR_FIGS_MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    from figs.utils.fonts import initialize_fonts
    initialize_fonts()

    FW_PANEL = "bold"
    FS_PANEL = 28

    RUN_PRESENTATION = args.figure == "presentation"

    RUN_DATA_FIG_1 = args.figure == "1"
    RUN_MODEL_FIG_2 = args.figure == "2"
    RUN_ANTIBIOTICS_FIG_3 = args.figure == "3"
    RUN_CONSORTIA_SIM_FIG_4 = args.figure == "4"
    RUN_CONSORTIA_EXP_FIG_5 = args.figure == "5"

    RUN_MODEL_TRAINING_DATASET_COMPARISON_FIG_S1 = args.figure == "s1"
    RUN_MODEL_SUPPLEMENTAL_FIG_S2 = args.figure == "s2"
    RUN_MODEL_INTRINSIC_DIMENSIONS_FIG_S3 = args.figure == "s3"
    RUN_ANTIBIOTICS_KYERI_TOP10F_FIG_S4 = args.figure == "s4"
    RUN_ANTIBIOTICS_KYERI_KEIO_FIG_S5 = args.figure == "s5"
    RUN_ANTIBIOTICS_KYERI_TIMER_FIG_S6 = args.figure == "s6"
    RUN_ANTIBIOTICS_CAROLYN_ALL_FIG_S7 = args.figure == "s7"
    RUN_ANTIBIOTICS_MODEL_COMPARISON_FIG_S8 = args.figure == "s8"
    RUN_CONSORTIA_SIM_SUPPLEMENTAL_OVERVIEW_FIG_S9 = args.figure == "s9"
    RUN_CONSORTIA_SIM_SUPPLEMENTAL_FIG_S10 = args.figure == "s10"
    RUN_CONSORTIA_EXP_FOCAL_COMMUNITIES_FIG_S11 = args.figure == "s11"
    RUN_CONSORTIA_EXP_SUPPLEMENTAL_FIG_S12 = args.figure == "s12"

    if RUN_PRESENTATION:
        from figs.presentation import main
        print("Running RUN_PRESENTATION")
        main()

    if RUN_DATA_FIG_1:
        print("Running RUN_DATA")
        from figs.data.main_beta import main
        main(
            fw_panel=FW_PANEL,
            fs_panel=FS_PANEL,
        )

    if RUN_MODEL_FIG_2:
        print("Running RUN_MODEL_FIG_2")
        from figs.model.main_beta import main
        main(
            fw_panel=FW_PANEL,
            fs_panel=FS_PANEL,
        )

    if RUN_ANTIBIOTICS_FIG_3:
        print("Running RUN_ANTIBIOTICS")
        from figs.antibiotics.main_gamma import main
        main(
            fw_panel=FW_PANEL,
            fs_panel=FS_PANEL,
        )

    if RUN_CONSORTIA_SIM_FIG_4:
        print("Running RUN_CONSORTIA_SIM_FIG_4")
        from figs.consortia.main_epsilon import main
        main(
            fw_panel=FW_PANEL,
            fs_panel=FS_PANEL,
        )

    if RUN_CONSORTIA_EXP_FIG_5:
        from figs.consortia_exp.main_epsilon import main
        print("Running RUN_CONSORTIA_EXP_MAIN")
        main(
            fw_panel=FW_PANEL,
            fs_panel=FS_PANEL,
        )

    if RUN_MODEL_TRAINING_DATASET_COMPARISON_FIG_S1:
        from figs.model.training_dataset_comparison import main
        print("Running RUN_MODEL_TRAINING_DATASET_COMPARISON")
        main()

    if RUN_MODEL_SUPPLEMENTAL_FIG_S2:
        print("Running RUN_MODEL_SUPPLEMENTAL_FIG_S2")
        from figs.model.supplemental import main
        main()

    if RUN_MODEL_INTRINSIC_DIMENSIONS_FIG_S3:
        from figs.model.plot_intrinsic_dimensions import main
        print("Running RUN_MODEL_INTRINSIC_DIMENSIONS_FIG_S3")
        main()

    if RUN_ANTIBIOTICS_KYERI_TOP10F_FIG_S4:
        from figs.antibiotics.kyeri_top10f import main
        print("Running RUN_ANTIBIOTICS_KYERI_TOP10F_FIG_S4")
        main()

    if RUN_ANTIBIOTICS_KYERI_KEIO_FIG_S5:
        from figs.antibiotics.kyeri_keio import main
        print("Running RUN_ANTIBIOTICS_KYERI_KEIO_FIG_S5")
        main(n_cols=8)

    if RUN_ANTIBIOTICS_KYERI_TIMER_FIG_S6:
        from figs.antibiotics.kyeri_timer import main
        print("Running RUN_ANTIBIOTICS_KYERI_TIMER_FIG_S6")
        main()

    if RUN_ANTIBIOTICS_CAROLYN_ALL_FIG_S7:
        from figs.antibiotics.carolyn_all import main
        print("Running RUN_ANTIBIOTICS_CAROLYN_ALL_FIG_S7")
        main()

    if RUN_ANTIBIOTICS_MODEL_COMPARISON_FIG_S8:
        print("Running RUN_ANTIBIOTICS_MODEL_COMPARISON_FIG_S8")
        from figs.antibiotics.plot_antibiotics_summary import main
        main(
            fs_panel=FS_PANEL,
            fw_panel=FW_PANEL,
        )

    if RUN_CONSORTIA_SIM_SUPPLEMENTAL_OVERVIEW_FIG_S9:
        from figs.consortia.supplemental_consortia_overview import main
        print("Running RUN_CONSORTIA_SIM_SUPPLEMENTAL_OVERVIEW_FIG_S9")
        main()

    if RUN_CONSORTIA_SIM_SUPPLEMENTAL_FIG_S10:
        print("Running RUN_CONSORTIA_SIM_SUPPLEMENTAL_FIG_S10")
        from figs.consortia.supplemental_curves import main
        main(
            fs_panel=FS_PANEL,
            fw_panel=FW_PANEL,
        )

    if RUN_CONSORTIA_EXP_FOCAL_COMMUNITIES_FIG_S11:
        print("Running RUN_CONSORTIA_EXP_FOCAL_COMMUNITIES_FIG_S11")
        from figs.consortia_exp.for_presentation import main
        main()
        
    if RUN_CONSORTIA_EXP_SUPPLEMENTAL_FIG_S12:
        print("Running RUN_CONSORTIA_EXP_SUPPLEMENTAL_FIG_S12")
        from figs.consortia_exp.supplemental_curves import main
        main(
            fs_panel=FS_PANEL,
            fw_panel=FW_PANEL,
        )
