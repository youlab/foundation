import matplotlib.pyplot as plt

from applications.antibiotics.kyeri.data import get_df
from applications.utils.data import get_t_cols
from config import DIR_FIGS_MANUSCRIPT
from figs.utils.colors import get_colors


def process_strain_category(
    strain,
):
    if (
        strain == "top10f"
    ) or (
        strain == "FJ_TIMER_MG1655_P15A"
    ):
        return strain
    return "Keio"


def format_axis(a):
    a.set_ylim(0, 1.2,)
    a.set_xlim(0, 127,)
    a.axis("off")


def correct_keio_strain(strain):
    return "Keio_" + strain.split("_")[-1]


def main(
    n_cols=12,
    w_total=1,
    p_x=0.001,
    p_y=0.001,
    fs_label=16,
    fw_label="semibold",
    fs_ticks=14,
    fs_text=8,
    fw_text="normal",
    y_text=0.08,
    x_text=0.28,
    x_space=0.13,
    fw_legend="medium",
    fs_legend=12,
):
    df = get_df()
    t_cols, t = get_t_cols(p=df)
    df["strain_category"] = [process_strain_category(strain=strain) for strain in df.Strain]
    df_keio = df.loc[df.strain_category == "Keio", :].copy()
    df_keio["Strain"] = [correct_keio_strain(strain=strain) for strain in df_keio.Strain]
    colors = get_colors(
        n=6,
    )
    colors = colors[1:]
    params = {
        'amoxicillin': {
            "color": colors[0],
            "ab_conc_max": 30,
        },
        'carbenicillin': {
            "color": colors[1],
            "ab_conc_max": 50,
        },
        'cefotaxime': {
            "color": colors[2],
            "ab_conc_max": 7,
        },
        'levofloxacin': {
            "color": colors[3],
            "ab_conc_max": 40,
        },
        'none': {
            "color": colors[4],
        },
    }

    n_panels = 96
    
    n_rows = n_panels // n_cols
    x_step = 1 / n_cols * w_total
    y_step = 1 / n_rows

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(n_cols, n_rows,),
    )
    ax.axis("off")

    idx_labels = (n_rows - 1) * n_cols

    for i, strain in enumerate(df_keio.Strain.unique()):
        i_col = i % n_cols
        i_row = n_rows - (i // n_cols) - 1
        a = ax.inset_axes(
            [i_col * x_step + p_x, i_row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
        )
        a.text(
            10,
            1.2,
            strain.replace("_strainID", ""),
            verticalalignment="top",
            fontsize=fs_text,
            fontweight=fw_text,
        )

        mask_strain = df_keio.Strain == strain
        for antibiotic_type in df_keio.loc[mask_strain, "antibiotic_type"].unique():
            mask_ab_type = mask_strain & (df_keio.antibiotic_type == antibiotic_type)
            for antibiotic_conc in df_keio.loc[mask_ab_type, "antibiotic_conc"].unique():
                mask_ab_conc = mask_ab_type & (df_keio.antibiotic_conc == antibiotic_conc)
                a.plot(
                    t,
                    df_keio.loc[mask_ab_conc, t_cols].to_numpy().T,
                    color=params[antibiotic_type]["color"],
                    alpha=(0.5 if (antibiotic_type == "none") else (antibiotic_conc / params[antibiotic_type]["ab_conc_max"])) * 0.8,
                )
        format_axis(a=a)

        if i == idx_labels:
            a.axis("on")
            for spine in ["top", "right",]:
                a.spines[spine].set_visible(False)
            a.set_yticks(
                [0, 1,],
                labels=[0, 1],
                fontsize=fs_ticks,
            )
            a.set_xticks(
                [0, 100,],
                labels=[0, 100,],
                fontsize=fs_ticks,
            )
            a.set_xlabel(
                "Time",
                fontsize=fs_label,
                fontweight=fw_label,
            )
            a.set_ylabel(
                "OD600",
                fontsize=fs_label,
                fontweight=fw_label,
            )

    for i, antibiotic_type in enumerate(params.keys()):
        fig.text(
            x_text + x_space * i,
            y_text,
            antibiotic_type,
            color=params[antibiotic_type]["color"],
            fontweight=fw_legend,
            fontsize=fs_legend,
        )


    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_5.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_5.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_5.svg",
    )
    plt.close()
