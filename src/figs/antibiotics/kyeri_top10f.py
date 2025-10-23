import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as cbar
import numpy as np

import numpy as np

from config import DIR_FIGS_MANUSCRIPT
from applications.utils.data import get_t_cols
from applications.antibiotics.kyeri.data import get_df


def process_dilfac(x):
    val = x.split("_")[-1]
    val = val.split("10^")
    if val[0] == "":
        y = 1
    else:
        y = float(val[0][:3])
    return np.around(y * 10 ** (-int(val[1][-1])), 10)


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

def main():
    fig, ax = plt.subplots(3, 5, figsize=(10, 7,),)
    ax = ax.ravel()

    fs_label = 16
    fs_ticks = 14
    fw_label = "medium"
    df = get_df()
    df["strain_category"] = [process_strain_category(strain=strain) for strain in df.Strain]
    df_top10f = df.loc[df.strain_category == "top10f", :].copy()
    t_cols, t = get_t_cols(p=df_top10f)

    df_top10f["dilution"] = np.log10([process_dilfac(x=val) for val in df_top10f.conditions_dilutionfactor])

    idx = df_top10f.groupby(
        [
            "media",
            "antibiotic_type",
            "antibiotic_conc",
        ]
    ).count().index

    dilutions = df_top10f.dilution.unique()

    cax = fig.add_axes([0.65, 0.15, 0.2, 0.05])

    cmap = plt.get_cmap('gist_earth_r')
    norm = mcolors.Normalize(vmin=min(dilutions), vmax=max(dilutions))

    cb = cbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Initial cell density (10^)')

    colors = [cmap(val) for val in np.linspace(0.2, 1, len(dilutions))]

    for i, (media, antibiotic_type, antibiotic_conc,) in enumerate(idx):
        print(i+1, media, antibiotic_type, antibiotic_conc)
        mask = (
            df_top10f.media == media
        ) & (
            df_top10f.antibiotic_type == antibiotic_type
        ) & (
            df_top10f.antibiotic_conc == antibiotic_conc
        )
        temp = df_top10f.loc[mask, :].copy()
        for j, dilution in enumerate(reversed(dilutions)):
            mask = temp.dilution == dilution
            ax[i].plot(
                t,
                temp.loc[mask, t_cols].to_numpy().T,
                color=colors[j],
                alpha=0.4,
            )
        ax[i].text(
            10,
            1,
            i + 1,
            verticalalignment="top",
            fontsize=16,
            fontweight="medium",
        )
        if i != 10:
            ax[i].set_yticks(
                [0, 0.5, 1,],
                labels=[],
            )
            ax[i].set_xticks(
                [0, 50, 100,],
                labels=[],
            )
        else:
            ax[i].set_yticks(
                [0, 0.5, 1,],
                labels=[0, 0.5, 1,],
                fontsize=fs_ticks,
            )
            ax[i].set_xticks(
                [0, 50, 100,],
                labels=[0, 50, 100,],
                fontsize=fs_ticks,
            )
            ax[i].set_xlabel(
                "Time",
                fontsize=fs_label,
                fontweight=fw_label,
            )
            ax[i].set_ylabel(
                "OD600",
                fontsize=fs_label,
                fontweight=fw_label,
            )

        ax[i].set_xlim(0, t.max())
        ax[i].set_ylim(0, 1,)

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    for a in ax[-2:]:
        a.axis("off")

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_4.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_4.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_4.svg",
    )
    plt.close()
