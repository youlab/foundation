import matplotlib.pyplot as plt

from applications.antibiotics.kyeri.data import get_df
from applications.utils.data import get_t_cols
from config import DIR_FIGS_MANUSCRIPT


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


def format_axis(
    a,
    y_max,
):
    a.set_ylim(0, y_max,)
    a.set_xlim(0, 127,)
    a.axis("off")


def main(
    fs_text=12,
    fw_text="medium",
    fs_ticks=12,
    fw_label="semibold",
    fs_label=14,
    y_text=0.05,
    x_text=0.6,
    x_space=0.12,
    fw_legend="medium",
    fs_legend=12,
):
    df = get_df()
    t_cols, t = get_t_cols(p=df)
    df["strain_category"] = [process_strain_category(strain=strain) for strain in df.Strain]
    df_timer = df.loc[df.strain_category == "FJ_TIMER_MG1655_P15A", :].copy()

    color_keys = {}
    for val in df_timer.groupby(
        [
            "antibiotic_type",
            "antibiotic_conc",
        ]
    ).count().index:
        if val[0] not in color_keys:
            color_keys[val[0]] = []
        color_keys[val[0]].append(val[1])

    keys = {
        "A": ('M9CA+0.4%glucose', 'carbenicillin', 0.0,),
        "B": ('M9CA+0.4%glucose', 'carbenicillin', 0.02,),
        "C": ('M9CA+0.4%glucose', 'carbenicillin', 0.1,),
        "D": ('M9CA+0.4%glucose', 'carbenicillin', 0.2,),
        "E": ('M9CA+0.4%glucose', 'cefotaxime', 0.02,),
        "F": ('M9CA+0.4%glucose', 'cefotaxime', 0.2,),
        "G": ('M9CA+0.4%glucose', 'amoxicillin', 0.02,),
        "H": ('M9CA+0.4%glucose', 'amoxicillin', 0.2,),
        "I": ('M9CA+0.4%glucose', 'carbenicillin/chloramphenicol', 0.2,),
        "J": ('M9CA+0.4%glucose', 'carbenicillin/chloramphenicol', 0.0,),
        "K": ('M9CA+0.4%glucose', 'chloramphenicol', 0.0,),
        "L": ('M9CA+0.4%glucose', 'chloramphenicol', 0.2,),
        "M": ('M9CA+0.4%glucose', 'none', 0.0,),
        "N": ('M9CA+0.4%glucose', 'none', 0.02,),
        "O": ('M9CA+0.4%glucose', 'none', 0.1,),
        "P": ('M9CA+0.4%glucose', 'none', 0.2,),
        "Q": ("LB",),
        "R": ("tbroth",),
    }

    fig = plt.figure(
        figsize=(8, 6,),
        tight_layout=True,
    )

    ax_dict = fig.subplot_mosaic(
        "AAABBBCCCDDD;EEEFFFGGGHHH;IIIJJJKKKLLL;MMNNOOPPQQRR",
        height_ratios=[1,1,1,0.6,],
    )

    dfs = {}
    for i_ax_label, ax_label in enumerate(ax_dict.keys()):
        key = keys[ax_label]
        media = key[0]
        mask = df_timer.media == media
        print(len(key), key)
        if len(key) > 1:
            antibiotic_type = key[1]
            mask = mask & (df_timer.antibiotic_type == antibiotic_type)
            if len(key) > 2:
                cas = key[2]
                mask = mask & (df_timer.cas == cas)
        else:
            antibiotic_type = "carbenicillin"
            cas = 0
        
        dfs[ax_label] = df_timer.loc[mask]
        
        reading = "sfGFP_70"
        mask_reading = mask & (df_timer.Reading == reading)
        ax1 = ax_dict[ax_label].twinx()
        for j, antibiotic_conc in enumerate(color_keys[antibiotic_type]):
            mask_conc = mask_reading & (df_timer.antibiotic_conc == antibiotic_conc)
            ax1.plot(
                df_timer.loc[mask_conc, t_cols].to_numpy().T,
                alpha=(j + 1) / 8,
                color="cornflowerblue",
            )
        
        format_axis(
            a=ax1,
            y_max=6_500,
        )

        reading = "dTimer"
        mask_reading = mask & (df_timer.Reading == reading)
        ax1 = ax_dict[ax_label].twinx()
        for j, antibiotic_conc in enumerate(color_keys[antibiotic_type]):
            mask_conc = mask_reading & (df_timer.antibiotic_conc == antibiotic_conc)
            ax1.plot(
                df_timer.loc[mask_conc, t_cols].to_numpy().T,
                alpha=(j + 1) / 8,
                color="firebrick",
            )
        
        format_axis(
            a=ax1,
            y_max=3_000,
        )
        
        reading = "OD600"
        mask_reading = mask & (df_timer.Reading == reading)
        format_axis(
            a=ax_dict[ax_label],
            y_max=0.8,
        )
        for j, antibiotic_conc in enumerate(color_keys[antibiotic_type]):
            mask_conc = mask_reading & (df_timer.antibiotic_conc == antibiotic_conc)
            ax_dict[ax_label].plot(
                df_timer.loc[mask_conc, t_cols].to_numpy().T,
                alpha=(j + 1) / 7,
                color="k",
            )
        
        ax_dict[ax_label].text(
            10,
            0.8,
            i_ax_label + 1,
            fontsize=fs_text,
            fontweight=fw_text,
            verticalalignment="top",
        )


    ax_dict["M"].axis("on")
    for spine in ["top", "right",]:
        ax_dict["M"].spines[spine].set_visible(False)
    ax_dict["M"].set_yticks(
        [],
    )
    ax_dict["M"].set_xticks(
        [0, 100,],
        labels=[0, 100,],
        fontsize=fs_ticks,
    )
    ax_dict["M"].set_xlabel(
        "Time",
        fontsize=fs_label,
        fontweight=fw_label,
    )
    ax_dict["M"].set_ylabel(
        "Reading",
        fontsize=fs_label,
        fontweight=fw_label,
    )
    for i, (reading, color,) in enumerate(
        [
            ("OD600", "k",),
            ("sfGFP_70", "cornflowerblue",),
            ("dTimer", "firebrick",),
        ]
    ):
        fig.text(
            x_text + x_space * i,
            y_text,
            reading,
            color=color,
            fontweight=fw_legend,
            fontsize=fs_legend,
        )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_6.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_6.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_6.svg",
    )
    plt.close()
