import numpy as np

from figs.utils.colors import get_colors
from data.normalization_functions.utils import get_t_cols
from applications.antibiotics.kyeri.data import get_df


def plot_kyeri_concentrations(
    ax,
    x_text=5,
    y_text=0.8,
    fs_text=14,
    fs_legend=14,
    fs_ticks=12,
    fs_label=14,
    lw=2,
    alpha=0.7,
    p_x=0.01,
    p_y=0.02,
    w_total=0.8,
):
    n_cols = 4
    x_step = 1 / n_cols * w_total

    a = []
    for i_col in range(n_cols):
        a.append(
            ax.inset_axes(
                [i_col * x_step + p_x, p_y, x_step - 2 * p_x, 1 - 2 * p_y,]
            )
        )
    ax = np.array(a)

    df = get_df()
    t_cols, _ = get_t_cols(p=df)
    docs = df.groupby(["date_of_collection", "cas", "antibiotic_type",]).count().reset_index().date_of_collection.unique()
    mask_doc = np.array([doc in docs for doc in df.date_of_collection])
    antibiotics = ["carbenicillin", "none",]
    mask_antibiotic = np.array([antibiotic in antibiotics for antibiotic in df.antibiotic_type])
    df = df.loc[mask_doc & mask_antibiotic].copy()
    print(df.Strain.unique())
    colors = get_colors(n=6)
    for i, cas_conc in enumerate([0, 0.02, 0.1, 0.2,]):
        for j, ab_conc in enumerate(df.antibiotic_conc.unique()):
            mask = (df.cas == cas_conc) & (df.antibiotic_conc == ab_conc)
            if mask.sum() == 0:
                continue
            if i == 3:
                legend_labels = [ab_conc]
                legend_labels.extend(["_nolegend"] * (mask.sum() - 1))
            else:
                legend_labels = ["_nolegend"] * mask.sum()
            ax[i].plot(
                df.loc[mask, t_cols].to_numpy().T,
                color=colors[j],
                lw=lw,
                alpha=alpha,
                label=legend_labels,
            )
        ax[i].text(
            x=x_text,
            y=y_text,
            s=f"{cas_conc}% casaa",
            fontsize=fs_text,
            horizontalalignment="left",
            verticalalignment="top",
        )
        for spine in ["top", "right",]:
            ax[i].spines[spine].set_visible(False)
        if i == 0:
            x_tick_labels = [0, 64, 128,]
            y_tick_labels = [0.0, 0.4, 0.8,]
            ax[i].set_ylabel(
                "Cell density",
                fontsize=fs_label,
            )
            ax[i].set_xlabel(
                "Time",
                fontsize=fs_label,
            )
        else:
            x_tick_labels = []
            y_tick_labels = []
            
        ax[i].set_xticks(
            [0, 64, 128,],
            labels=x_tick_labels,
            fontsize=fs_ticks,
        )
        ax[i].set_yticks(
            [0, 0.4, 0.8,],
            labels=y_tick_labels,
            fontsize=fs_ticks,
        )
        ax[i].set_xticks(
            [32, 96,],
            minor=True,
        )
        ax[i].set_yticks(
            [0.1, 0.2, 0.3, 0.5, 0.6, 0.7,],
            minor=True,
        )
        ax[i].set_ylim(0, 0.8,)
        ax[i].set_xlim(0, 128,)
    ax[3].legend(
        title=r"Carbenicillin ($\mu$g/mL)",
        bbox_to_anchor=(1.05, 1,),
        fontsize=fs_legend,
        title_fontsize=fs_legend,   
    )
