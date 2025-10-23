import numpy as np
from figs.utils.colors import get_colors
from applications.antibiotics.carolyn.data import get_data


def plot_strains(
    ax,
    df,
    strains,
    antibiotics=None,
    lw=2,
    alpha=0.8,
    p_x=0.01,
    p_y=0.01,
    w_total=1,
    n_panels=4,
    n_cols=2,
    x_text=5,
    y_text=0.6,
    fs_text=12,
    fs_ticks=10,
    fs_legend=12,
    xlim=(0, 100),
    ylim=(0, 0.6,),
    add_legend=False,
    colors=get_colors(n=4),
):
    ax.axis("off")
    n_rows = n_panels // n_cols
    x_step = 1 / n_cols * w_total
    y_step = 1 / n_rows
    
    for i in range(4):
        i_col = i % n_cols
        i_row = i // n_cols
        mask_strain = df.strain == strains[i]
        a = ax.inset_axes(
            [i_col * x_step + p_x, i_row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
        )
        for i_condition in range(4):
            a.plot(
                df.loc[mask_strain, 0:98].iloc[i_condition * 4:i_condition * 4 + 4].to_numpy().T,
                color=colors[i_condition],
                label=[f"Condition {i_condition + 1}", "_nolegend", "_nolegend", "_nolegend",],
                lw=lw,
                alpha=alpha,
            )
        if antibiotics is not None:
            a.text(
                x=x_text,
                y=y_text,
                s=f"{antibiotics[i]} resistant",
                fontsize=fs_text,
                verticalalignment="center",
            )
        if (i_col == 0) and (i_row == 0) and (not add_legend):
            a.set_xticks(
                [0, 50, 100,],
                labels=[0, 50, 100,],
                fontsize=fs_ticks,
            )
            a.set_yticks(
                [0.0, 0.3, 0.6,],
                labels=[0.0, 0.3, 0.6,],
                fontsize=fs_ticks,
            )
            a.set_xlabel(
                "Time",
                fontsize=fs_text,
            )
            a.set_ylabel(
                "Cell density",
                fontsize=fs_text,
            )
        else:
            a.set_xticks(
                [0, 50, 100,],
                labels=[],
            )
            a.set_yticks(
                [0.0, 0.3, 0.6,],
                labels=[],
            )
        if (i_col == 1) and (i_row == 1) and add_legend:
            a.legend(
                bbox_to_anchor=(1.05, 1),
                fontsize=fs_legend,
            )

        a.set_xlim(xlim)
        a.set_ylim(ylim)
        for spine in ["top", "right",]:
            a.spines[spine].set_visible(False)


def plot_carolyn_examples(
    ax,
    p_x=0.1,
    p_y=0.1,
    w_total=0.9,
):
    n_cols = 2
    n_panels = 2
    x_step = 1 / n_cols * w_total

    a = []
    for i_col in range(2):
        a.append(
            ax.inset_axes(
                [i_col * x_step + p_x, p_y, x_step - 2 * p_x, 1 - 2 * p_y,]
            )
        )
    ax = np.array(a)

    df = get_data()
    antibiotics = ["SAM", "GM", "SXT", "CIP",]
    mask_only_one = df.loc[:, antibiotics].sum(axis=1) == 1
    mask_all_four = df.loc[:, antibiotics].sum(axis=1) == 4
    strains_of_interest_only_one = [
        df.loc[mask_only_one & (df.loc[:, ab] == 1)].strain.iloc[0] for ab in antibiotics
    ]

    strains_of_interest_all_four = [
        df.loc[mask_all_four].strain.iloc[i * 50] for i in range(4)
    ]

    plot_strains(
        ax=ax[0],
        df=df,
        strains=strains_of_interest_only_one,
        antibiotics=antibiotics,
        p_x=0.05,
        p_y=0.05,
    )

    plot_strains(
        ax=ax[1],
        df=df,
        strains=strains_of_interest_all_four,
        add_legend=True,
        p_x=0.05,
        p_y=0.05,
    )