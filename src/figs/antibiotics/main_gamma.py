import matplotlib.pyplot as plt

from config import (
    DIR_FIGS_MANUSCRIPT,
    MODEL_TYPE,
    Z_DIM,
)
from figs.antibiotics.antibiotic_classification_pipeline import plot_antibiotic_classification_pipeline
from figs.antibiotics.antibiotic_regression_pipeline import plot_antibiotic_regression_pipeline
from figs.antibiotics.classify_antibiotic_resistance_f1_score import plot_resistance_panels
from figs.antibiotics.classify_antibiotic_treatment import main as plot_classify_antibiotic_treatment
from figs.antibiotics.regress_antibiotic_concentration import main as plot_regress_antibiotic_concentration
from figs.antibiotics.resistance_classification_pipeline import plot_resistance_classification_pipeline

FS_LABELS = 16
FS_TICKS = 14
FS_TEXT = 16


def format_axes_for_kyeri(
    ax,
    xticks_major,
    xticks_minor,
    yticks_major,
    yticks_minor,
    xlabel,
    ylabel,
    ylim=None,
):
    ax.set_xticks(
        xticks_major,
        labels=[f"{val:,.0f}" for val in xticks_major],
        fontsize=FS_TICKS,
    )
    ax.set_xticks(
        xticks_minor,
        minor=True,
    )
    ax.set_yticks(
        yticks_major,
        labels=yticks_major,
        fontsize=FS_TICKS,
    )
    ax.set_yticks(
        yticks_minor,
        minor=True,
    )
    ax.set_xlabel(
        xlabel,
        fontsize=FS_LABELS,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=FS_LABELS,
    )
    if ylim is not None:
        ax.set_ylim(ylim)

    
def main(
    fs_panel,
    fw_panel,
):
    fig = plt.figure(
        figsize=(16, 9,),
        layout="tight",
    )
    ax_dict = fig.subplot_mosaic(
        "A",
    )

    metric_labels = [
        "raw_accuracy",
        "latent_accuracy",
    ]

    for key in ax_dict.keys():
        ax_dict[key].axis("off")
    w_pipeline = 0.4
    w_panel = 0.15
    h = 0.3
    p = 0.08
    b1, b2, b3 = 0.71, 0.38, 0.05
    a_clas = ax_dict["A"].inset_axes([p - 0.03, b1, w_pipeline, h,])
    p_clas = ax_dict["A"].inset_axes([w_pipeline + p * 2 - 0.03, b1, w_panel, h * 0.9,])
    a_regr = ax_dict["A"].inset_axes([p - 0.03, b2, w_pipeline, h,])
    p_regr = ax_dict["A"].inset_axes([w_pipeline + p * 2 - 0.03, b2, w_panel, h * 0.9,])
    
    a_resi = ax_dict["A"].inset_axes([p - 0.03, b3, w_pipeline, h,])
    axes_for_resistance = []
    ys = [0.45, 0.45, 0.05, 0.05,]
    h, w_panel = 0.1, 0.06
    xes = [0, w_panel + p /2, 0, w_panel + p / 2,]
    
    for i in range(4):
        axes_for_resistance.append(
            ax_dict["A"].inset_axes(
                [
                    xes[i] + w_pipeline + p * 2 - 0.03,
                    b3 + ys[i] / 3,
                    w_panel,
                    h,
                ]
            )
        )
    plot_resistance_classification_pipeline(
        ax=a_resi,
        w_bact=0.22,
        h_bact=0.2,
        y_bact_hi=0.7,
        y_bact_lo=0.15,
        x_bact=0.75,
    )
    p = 0.05
    xes = [0, 0.25, 0.5, 0.75,]
    ys = [0.05, 0.05, 0.05, 0.05,]
    h, w = 0.9, 0.23

    plot_resistance_panels(
        axes=axes_for_resistance,
        model_name=f"{MODEL_TYPE}_{Z_DIM}",
        metric_labels=metric_labels,
    )
    
    plot_antibiotic_classification_pipeline(
        ax=a_clas,
        x_adder=0.3,
    )

    plot_antibiotic_regression_pipeline(
        ax=a_regr,
        h_box=0.3,
        w_box=0.6,
        p_right=0.15,
    )

    xes = [0.15, 0.65]
    axes_for_kyeri = [p_clas, p_regr]
    for i in range(2):
        for spine in ["top", "right",]:
            axes_for_kyeri[i].spines[spine].set_visible(False)

    plot_classify_antibiotic_treatment(
        model_name=f"{MODEL_TYPE}_{Z_DIM}",
        metric_labels=metric_labels,
        ax=axes_for_kyeri[0],
        add_formatting=False,
    )

    plot_regress_antibiotic_concentration(
        model_name=f"{MODEL_TYPE}_{Z_DIM}",
        metric_labels=metric_labels,
        ax=axes_for_kyeri[1],
        add_formatting=False,
    )

    format_axes_for_kyeri(
        ax=axes_for_kyeri[0],
        xticks_major=[0, 2000, 4000],
        xticks_minor=[500, 1000, 1500, 2500, 3000, 3500,],
        yticks_major=[0.6, 0.7, 0.8, 0.9,],
        yticks_minor=[0.55, 0.65, 0.75, 0.85,],
        xlabel="Train size",
        ylabel="Classification accuracy",
    )

    format_axes_for_kyeri(
        ax=axes_for_kyeri[1],
        xticks_major=[0, 2500, 5000],
        xticks_minor=[500, 1000, 1500, 2000, 3000, 3500, 4000, 4500,],
        yticks_major=[0.4, 0.6, 0.8,],
        yticks_minor=[0.5, 0.7,],
        xlabel="Train size",
        ylabel="Regression accuracy",
        ylim=(0.35, 0.85,),
    )

    for x, y, label in [
        (0.02, 0.97, "A",),
        (0.02, 0.65, "C",),
        (0.02, 0.33, "E",),
        (0.46, 0.97, "B",),
        (0.46, 0.65, "D",),
        (0.46, 0.33, "F",),
    ]:
        fig.text(
            x=x,
            y=y,
            s=label,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=fs_panel,
            fontweight=fw_panel,
        )
    
    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_3.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_3.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "fig_3.svg",
    )
    plt.close()
