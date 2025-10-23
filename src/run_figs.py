if __name__ == "__main__":
    from figs.utils.fonts import initialize_fonts
    initialize_fonts()

    FW_PANEL = "bold"
    FS_PANEL = 28

    RUN_PRESENTATION = False 

    RUN_DATA_FIG_1 = False  
    RUN_MODEL_FIG_2 = True
    RUN_ANTIBIOTICS_FIG_3 = False
    RUN_CONSORTIA_SIM_FIG_4 = False
    RUN_CONSORTIA_EXP_FIG_5 = False

    RUN_MODEL_TRAINING_DATASET_COMPARISON_FIG_S1 = False 
    RUN_MODEL_SUPPLEMENTAL_FIG_S2 = False 
    RUN_MODEL_INTRINSIC_DIMENSIONS_FIG_S3 = False  
    RUN_ANTIBIOTICS_KYERI_TOP10F_FIG_S4 = False  
    RUN_ANTIBIOTICS_KYERI_KEIO_FIG_S5 = False 
    RUN_ANTIBIOTICS_KYERI_TIMER_FIG_S6 = False  
    RUN_ANTIBIOTICS_CAROLYN_ALL_FIG_S7 = False  
    RUN_ANTIBIOTICS_MODEL_COMPARISON_FIG_S8 = False  
    RUN_CONSORTIA_SIM_SUPPLEMENTAL_OVERVIEW_FIG_S9 = False  
    RUN_CONSORTIA_SIM_SUPPLEMENTAL_FIG_S10 = False  
    RUN_CONSORTIA_EXP_FOCAL_COMMUNITIES_FIG_S11 = False  
    RUN_CONSORTIA_EXP_SUPPLEMENTAL_FIG_S12 = False 

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
