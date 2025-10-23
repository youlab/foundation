import matplotlib.pyplot as plt
import numpy as np

from applications.config import NOW_TEXT
from config import (
    DIR_FIGS_PRESENTATION,
    DIR_RESULTS_CONSORTIA_EXP,
)

FS_LABEL = 20
FS_TICKS = 16
FS_TEXT = 20
FS_AB = 14
FS_LEGEND = 14


def get_ax(
    figsize=(10, 5,),
    axis_off=True,
):
    fig = plt.figure(
        figsize=figsize,
    )
    ax = fig.subplot_mosaic(
        "A",
    )["A"]
    if axis_off:
        ax.axis("off")
    return ax


def save(file_name):
    plt.savefig(
        DIR_FIGS_PRESENTATION
        / f"{file_name}.png",
    )
    plt.close()


def main():
    from figs.data.animation import main
    main()
    # print("Plotting intro")
    # intro()
    # print("Plotting fig 2")
    # fig2()
    # # print("Plotting fig 3")
    # fig3()
    # print("Plotting fig 4")
    # fig4()
    # print("Plotting fig 5")
    # fig5()
    # print("Plotting fig1 extra")
    # fig1_extra()
    # print("Plotting fig3 extra")
    # fig3_extra()
    # print("Plotting fig4 extra")
    # fig4_extra()
    # print("Plotting fig3 extra")
    # fig3_extra()


def intro():
    from figs.introduction.growth_curves import plot_growth_curves
    ax = get_ax()
    ax = [
        ax.inset_axes(
            [0.05, 0.1, 0.45, 0.8,],
        ),
        ax.inset_axes(
            [0.55, 0.1, 0.45, 0.8,],
        ),
    ]
    plot_growth_curves(
        ax=ax,
        lw=5,
        fs_label=20,
    )
    save(file_name="0A")

    from figs.introduction.specific_task_architecture import main
    main(use_fm_output=True)
    save(file_name="0B1")

    main(use_fm_output=True)
    save(file_name="0B2")

    from figs.introduction.plausible_shapes import main
    main()
    save(file_name="0C")


def fig1_extra():
    from figs.data.truncate_and_normalize import main
    main()
    save(file_name="1x0")

    from figs.data.interpolation import main
    ax = get_ax()
    a = []
    w = 0.8
    h = 0.28
    for bounds in [
        [(1-w)/2, 0, w, h,],
        [(1-w)/2, 0.33, w, h,],
        [(1-w)/2, 0.67, w, h,],
    ]:
        a.append(
            ax.inset_axes(
                bounds,
            )
        )
    main(ax=ax)
    save(file_name="1x1")


def fig3_extra():
    from figs.antibiotics.data_sparsity import main
    ax = get_ax()
    main(ax=ax)
    save(file_name="3x0")

    from figs.antibiotics.carolyn_examples import plot_carolyn_examples
    ax = get_ax()
    plot_carolyn_examples(
        ax=ax,
        p_x=0.01,
        p_y=0.1,
    )
    save(file_name="3x1")

    from figs.antibiotics.kyeri_treatments import plot_kyeri_treatments
    ax = get_ax(figsize=(14, 5,),)
    plot_kyeri_treatments(
        ax=ax,
        n_cols=8,
        n_panels=96,
        lw=1,
        y_max=0.7,
        alpha=0.9,
        fs_legend=12,
        p_x=0.005,
        p_y=0.005,
    )
    save(file_name="3x2")

    from figs.antibiotics.kyeri_concentrations import plot_kyeri_concentrations
    ax = get_ax()
    plot_kyeri_concentrations(
        ax=ax,
        fs_label=18,
        p_x=0.02,
        p_y=0.15,
        w_total=0.85,
    )
    save(file_name="3x3")


def fig4_extra():
    from figs.consortia.simple_complex import main
    ax = get_ax()
    main(ax=ax)
    save(file_name="4x-2")


    from figs.consortia.sample_focal_community import plot_sample_focal_community
    ax = get_ax()
    plot_sample_focal_community(
        ax=ax.inset_axes(
            [
                0.1,
                0.1,
                0.8,
                0.8,
            ]
        ),
        fs_label=18,
        fs_ticks=16,
    )
    save(file_name="4x-1")
    
    from figs.consortia.simulated_consortia_counts import plot_simulated_consortia_counts
    ax = get_ax()
    plot_simulated_consortia_counts(
        ax=ax.inset_axes(
            [
                0.3,
                0,
                0.4,
                1,
            ]
        )
    )
    save(file_name="4x0")
    from figs.consortia.input_to_prediction import plot_input_to_prediction
    ax = get_ax(axis_off=False)
    plot_input_to_prediction(
        ax=ax,
    )
    save(file_name="4x1")

    from figs.consortia.estimated_intrinsic_dimensions import plot_consortia_estimated_intrinsic_dimensions
    ax = get_ax(axis_off=False)
    plot_consortia_estimated_intrinsic_dimensions(
        ax=ax,
    )
    save(file_name="4x2")


def fig5_extra():
    from figs.consortia_exp.for_presentation import (
        plot_strains_in_all_eight,
        plot_focal_counts,
    )
    _, ax = plt.subplots(
        4,
        2,
        figsize=(10, 5,),
        tight_layout=True,
    )
    ax = ax.ravel()
    plot_strains_in_all_eight(
        ax=ax,
    )
    save(file_name="5x1")

    ax = get_ax()
    ax = ax.inset_axes(
        [0.1, 0.1, 0.8, 0.8,],
    )
    plot_focal_counts(
        ax,
        bar_height = 0.8,
        fs_labels = 14,
        fs_legend = 14,
        fs_data_labels = 12,
    )
    save(file_name="5x2")


def fig1():
    from figs.data.data_sources import plot_data_sources
    from figs.data.three_types import plot_three_types
    ax = get_ax()
    a = []
    w = 0.22
    h = 0.5
    for bounds in [
        [0, (1-h)/2, w, h,],
        [0.33, (1-h)/2, w, h,],
        [0.67, (1-h)/2, w, h,],
    ]:
        a.append(
            ax.inset_axes(
                bounds,
            )
        )
    plot_three_types(
        ax=a,
        colormap_name="Blues",
        vertical=False,
    )
    save(file_name="1ABC")

    ax = get_ax()
    plot_data_sources(
        ax=ax,
        p_y=0.02,
        p_x=0.01,
        n_cols=8,
        show_numbers=False,
    )
    save(file_name="1D")


def fig2():
    from figs.model.reconstruction_pipeline import plot_reconstruction_pipeline
    from data.utils import get_data
    from ml.utils.load_models import load_default_model

    model = load_default_model()
    data, idx = get_data(category="all")

    ax = get_ax()
    ax = ax.inset_axes([0, 0.1, 1, 0.8,])
    plot_reconstruction_pipeline(
        ax=ax,
        data=data,
        idx=idx,
        model=model,
        fs_labels=22,
        p_x=20,
        p_y=0.15,
        fs_vae_label=16,
        lw=5,
        ax_bounds_1=[0.05, 0.1, 0.3, 0.8,],
        ax_bounds_2=[0.68, 0.1, 0.3, 0.8,],
        label_shapes=True,
        label_sizes=True,
    )
    save(file_name="2A")

    from figs.model.examples_beta import plot_examples_beta
    from figs.model.model_comparison import plot_model_comparison
    from figs.model.model_reconstruction_accuracy import plot_model_reconstruction_accuracy
    from figs.model.twod_by_source import plot_twod_by_source
    
    ax = get_ax()
    plot_model_reconstruction_accuracy(
        ax=ax,
        fs_ticks=18,
        fs_label=20,
        p_x=0.05,
        p_y=0.05,
        x_r2=0.1,
        y_r2=0.9,
        fs_r2=18,
        horizontal_alignment="left",
        x_dataset=0.6,
        use_cache=True,
    )
    save(file_name="2B")

    ax = get_ax()
    ax = ax.inset_axes([0.15, 0, 0.7, 1,])
    plot_twod_by_source(
        ax=ax,
        fs_text= 16,
        x_text=190,
        y_text=-140,
        vertical_alignment="center",
        horizontal_alignment="center",
        p_x=0.08,
        p_y=0.01,
        left=-0.05,
    )
    save(file_name="2C")

    ax = get_ax()
    ax = ax.inset_axes([0.15, 0.1, 0.7, 0.8,])
    plot_model_comparison(
        ax=ax,
        y_label=r"Reconstruction accuracy ($R^2$)",
    )
    save(file_name="2D")

    ax = get_ax()
    ax = ax.inset_axes([-0.1, 0.15, 1.2, 0.7,])
    plot_examples_beta(
        ax=ax,
        data=data,
        idx=idx,
        model=model,
    )
    save(file_name="2E")


def fig3():
    from config import (
        MODEL_TYPE,
        Z_DIM,
    )
    model_name = f"{MODEL_TYPE}_{Z_DIM}"
    metric_labels = [
        "raw_accuracy",
        "latent_accuracy",
    ]
    
    from figs.antibiotics.increasing_sparsity import plot_increasing_sparsity
    ax = get_ax()
    plot_increasing_sparsity(
        ax=ax,
        text_angle=15,
        fs_label=FS_LABEL,
    )
    save(file_name="3A")

    
    from figs.antibiotics.resistance_classification_pipeline import plot_resistance_classification_pipeline
    ax = get_ax()
    plot_resistance_classification_pipeline(
        ax=ax,
        w_bact=0.3,
        h_bact=0.2,
        y_bact_hi=0.7,
        y_bact_lo=0.15,
        x_bact=0.8,
        fs_label=FS_LABEL,
    )
    save(file_name="3B")


    from figs.antibiotics.classify_antibiotic_resistance_f1_score import plot_resistance_panels
    ax = get_ax(figsize=(12, 3.8))
    p_x = 0.06
    p_y = 0.3
    axes_for_resistance = []
    for i in range(4):
        axes_for_resistance.append(
            ax.inset_axes(
                [i * 0.25 + p_x / 2, p_y / 2, 0.25 - p_x, 1 - p_y,],
            )
        )
    plot_resistance_panels(
        axes=axes_for_resistance,
        model_name=model_name,
        metric_labels=metric_labels,
        add_legend=[False, False, False, True,],
        fs_ticks=FS_TICKS,
        fs_labels=FS_LABEL,
        fs_text=FS_TEXT,
    )
    save(file_name="3C")

    from figs.antibiotics.antibiotic_classification_pipeline import plot_antibiotic_classification_pipeline
    ax = get_ax()
    plot_antibiotic_classification_pipeline(
        ax=ax,
        x_adder=0.3,
        fs_label=FS_LABEL,
        fs_ab=FS_AB,
    )
    save(file_name="3D")

    from figs.antibiotics.antibiotic_regression_pipeline import plot_antibiotic_regression_pipeline
    ax = get_ax()
    plot_antibiotic_regression_pipeline(
        ax=ax,
        h_box=0.3,
        w_box=0.8,
        p_right=0.18,
        fs_label=FS_LABEL,
    )
    save(file_name="3E")

    from figs.antibiotics.classify_antibiotic_treatment import main as plot_classify_antibiotic_treatment
    from figs.antibiotics.regress_antibiotic_concentration import main as plot_regress_antibiotic_concentration
    
    from figs.antibiotics.main_gamma import format_axes_for_kyeri

    ax = get_ax()
    ax = ax.inset_axes([0.1, 0.1, 0.8, 0.8,])
    plot_classify_antibiotic_treatment(
        model_name=model_name,
        metric_labels=metric_labels,
        ax=ax,
        add_formatting=False,
        add_legend=False,
        fs_labels=FS_LABEL,
        fs_legend=FS_LEGEND,
        fs_ticks=FS_TICKS,
    )
    format_axes_for_kyeri(
        ax=ax,
        xticks_major=[0, 2000, 4000],
        xticks_minor=[500, 1000, 1500, 2500, 3000, 3500,],
        yticks_major=[0.6, 0.7, 0.8, 0.9,],
        yticks_minor=[0.55, 0.65, 0.75, 0.85,],
        xlabel="Train size",
        ylabel="Classification accuracy",
    )
    save(file_name="3F")

    ax = get_ax()
    ax = ax.inset_axes([0.1, 0.1, 0.8, 0.8,])
    plot_regress_antibiotic_concentration(
        model_name=model_name,
        metric_labels=metric_labels,
        ax=ax,
        add_formatting=False,
        add_legend=False,
        fs_labels=FS_LABEL,
        fs_legend=FS_LEGEND,
        fs_ticks=FS_TICKS,
        add_text=True,
    )
    
    format_axes_for_kyeri(
        ax=ax,
        xticks_major=[0, 2500, 5000],
        xticks_minor=[500, 1000, 1500, 2000, 3000, 3500, 4000, 4500,],
        yticks_major=[0.4, 0.6, 0.8,],
        yticks_minor=[0.5, 0.7,],
        xlabel="Train size",
        ylabel="Regression accuracy",
        ylim=(0.35, 0.85,),
    )
    save(file_name="3G")


def fig4():
    from figs.consortia.main_epsilon import (
        initialize,
        plot_forecasts,
        plot_pipeline,
        show_sliding_window,
    )
    df, consortia, data = initialize()

    ax = get_ax()
    ax = ax.inset_axes([0.0, 0.2, 1, 0.6,])
    plot_pipeline(
        a=ax,
        data=data,
    )
    save(file_name="4A")

    ax = get_ax()
    ax = ax.inset_axes([0.05, 0.1, 0.8, 0.9,])
    show_sliding_window(
        a=ax,
        y=data["w_test"][0, :, :].T,
    )
    save(file_name="4B")

    for i, label in enumerate(["C", "D",]):
        ax = get_ax()
        ax = ax.inset_axes([0, 0.1, 0.9, 0.9,])
        if i == 0:
            add_label_to_plot = (
                160,
                0.4,
                32,
                0.2,
            )
        elif i == 1:
            add_label_to_plot = (
                160,
                0.95,
                32,
                0.2,
            )

        plot_forecasts(
            ax=ax,
            df=df,
            consortium=consortia[i],
            p_y=0.02,
            p_x=0.01,
            side_legend=True,
            fs_legend=10,
            legend_cols=1,
            legend_loc="upper left",
            fs_accuracy=14,
            train_size=12800,
            accuracy_plot_right_pad=1,
            add_label_to_plot=add_label_to_plot,
        )
        save(file_name=f"4{label}")


def fig5():
    from figs.consortia_exp.experiment_conditions import plot_experiment_conditions
    ax = get_ax()
    ax = ax.inset_axes([0, 0.2, 0.8, 0.6,])
    plot_experiment_conditions(
        a=ax,
        fs1=18,
        fs2=18,
        fs_symbol=36,
        fs_text=24,
    )
    save(file_name="5A")

    from figs.consortia_exp.main_epsilon import plot_cv
    ax = get_ax()
    ax = ax.inset_axes([0, 0.2, 0.8, 0.6,])
    plot_cv(
        ax=ax,
        fs_number=FS_TICKS,
        fs_text=FS_TEXT,
        w_train=90,
        w_vert=60,
    )
    save(file_name="5B")

    experiment = "water-a"
    rep = "3"
    data = np.load(
        DIR_RESULTS_CONSORTIA_EXP
        / f"future_outlook_{experiment}-{rep}_latent_latent_{NOW_TEXT}.npz",
    )
    
    from figs.consortia.pipeline import plot_pipeline
    ax = get_ax()
    ax = ax.inset_axes([0, 0.3, 1, 0.4,])
    plot_pipeline(
        a=ax,
        data=data,
        alpha=0.8,
        y_label=1.1,
        fs_text=FS_TEXT,
    )
    save(file_name="5C")

    from figs.consortia_exp.main_epsilon import plot_forecasts
    ax = get_ax(figsize=(16, 5,))
    ax = ax.inset_axes([0, 0.1, 0.9, 0.9,])
    plot_forecasts(
        ax=ax,
        data=data,
        experiment=experiment,
        p_y=0.02,
        p_x=0.01,
        fs_legend=FS_LEGEND,
        side_legend=True,
    )
    save(file_name="5D")

    from figs.consortia_exp.plot_forecast_rmse import main as plot_forecast_rmse
    ax = get_ax(figsize=(7, 5,),)
    ax = ax.inset_axes([0.1, 0.1, 0.8, 0.8,])
    plot_forecast_rmse(
        ax=ax,
    )
    save(file_name="5E")

    from figs.utils.pipelines import plot_classification_pipeline
    ax = get_ax()
    ax = ax.inset_axes([0, 0.2, 0.8, 0.6,])
    plot_classification_pipeline(
        a=ax,
        data=data,
        w=50,
        fs_text=FS_TEXT,
    )
    save(file_name="5F")

    from figs.consortia_exp.plot_abundance_classification import main as plot_abundance_classification
    ax = get_ax(figsize=(7, 5,),)
    ax = ax.inset_axes([0.15, 0.15, 0.7, 0.7,])
    plot_abundance_classification(
        ax=ax,
    )
    save(file_name="5G")
