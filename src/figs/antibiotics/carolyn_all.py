import matplotlib.pyplot as plt

from applications.antibiotics.carolyn.data import get_data
from config import DIR_FIGS_MANUSCRIPT
from figs.utils.colors import get_colors


def calc_number_of_strains_resistant():
    from itertools import combinations
    
    df = get_data()
    
    all_cases = [(None,)]
    
    for i in range(1, 5):
        all_cases.extend(list(combinations(["SAM", "GM", "SXT", "CIP",], i)))
    
    out = {}
    for case in all_cases:
        params = {
            "SAM": 0,
            "GM": 0,
            "SXT": 0,
            "CIP": 0,
        }
        if case[0] is None:
            pass
        else:
            for ab in case:
                params[ab] = 1

        mask = df.loc[:, "SAM"] == params["SAM"]
        mask = mask & (df.loc[:, "GM"] == params["GM"])
        mask = mask & (df.loc[:, "SXT"] == params["SXT"])
        mask = mask & (df.loc[:, "CIP"] == params["CIP"])
        out[case] = mask.sum()
    
    numbers = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }
    
    for key in out.keys():
        if key[0] is None:
            numbers[0] = out[key]
        elif len(key) == 1:
            numbers[1] += out[key]
        elif len(key) == 2:
            numbers[2] += out[key]
        elif len(key) == 3:
            numbers[3] += out[key]
        elif len(key) == 4:
            numbers[4] += out[key]
    
    for key in numbers.keys():
        numbers[key] /= 16
    print(numbers)


def plot_strains(
    df,
    lw=1,
    alpha=0.8,
    p_x=0.01,
    p_y=0.01,
    w_total=1,
    n_panels=280,
    n_cols=14,
    fs_text=12,
    fs_ticks=10,
    xlim=(0, 100),
    ylim=(0, 1,),
    colors=get_colors(n=4),
    yfig_text=0.08,
    xfig_text=0.25,
    xfig_space=0.12,
    fw_legend="medium",
    fs_legend=12,
    fs_strain=8,
    fw_strain="normal",
):
    n_rows = n_panels // n_cols
    x_step = 1 / n_cols * w_total
    y_step = 1 / n_rows
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(
            n_cols,
            n_rows,
        ),
    )
    ax.axis("off")
    
    for i, strain in enumerate(sorted(df.strain.unique())):
        i_col = i % n_cols
        i_row = n_rows - i // n_cols - 1
        mask_strain = df.strain == strain
        a = ax.inset_axes(
            [i_col * x_step + p_x, i_row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
        )
        for i_condition in range(4):
            a.plot(
                df.loc[mask_strain, 0:98].iloc[i_condition * 4:i_condition * 4 + 4].to_numpy().T,
                color=colors[i_condition],
                lw=lw,
                alpha=alpha,
            )
        a.text(
            x=5,
            y=ylim[1],
            s=strain,
            verticalalignment="top",
            fontsize=fs_strain,
            fontweight=fw_strain,
        )
        if (i_col == 0) and (i_row == 0):
            a.set_xticks(
                [0, 100,],
                labels=[0, 100,],
                fontsize=fs_ticks,
            )
            a.set_yticks(
                [0.0, ylim[1],],
                labels=[0.0, ylim[1],],
                fontsize=fs_ticks,
            )
            a.set_xlabel(
                "Time",
                fontsize=fs_text,
            )
            a.set_ylabel(
                "OD600",
                fontsize=fs_text,
            )
            for spine in ["top", "right",]:
                a.spines[spine].set_visible(False)

        else:
            a.axis("off")

        a.set_xlim(xlim)
        a.set_ylim(ylim)
    for i, condition in enumerate(["10,000x\ndilution", "phage", "100x\ndilution", "carbenicillin"]):
        fig.text(
            xfig_text + xfig_space * i,
            yfig_text,
            condition,
            color=colors[i],
            fontweight=fw_legend,
            fontsize=fs_legend,
            verticalalignment="center",
            horizontalalignment="center",
        )


def main():
    df = get_data()
    plot_strains(
        df=df,
        p_x=0.005,
        p_y=0.005,
        alpha=0.5,
        ylim=(0, 0.8,),
        fs_legend=14,
        fw_legend="semibold",
        xfig_text=0.5,
        yfig_text=0.1,
        xfig_space=0.1,
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_7.png",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_7.pdf",
    )

    plt.savefig(
        DIR_FIGS_MANUSCRIPT
        / "supp_fig_7.svg",
    )
    plt.close()
