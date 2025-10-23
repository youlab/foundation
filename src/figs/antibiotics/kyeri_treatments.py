import numpy as np
from figs.utils.colors import get_colors
from data.normalization_functions.utils import get_t_cols
from applications.antibiotics.kyeri.data import get_df


def plot_kyeri_treatments(
    ax,
    lw = 4,
    alpha = 0.7,
    colors=get_colors(n=3),
    n_cols=12,
    p_x=0.01,
    p_y=0.01,
    n_panels=24,
    y_max=1,
    fs_label=12,
    fs_ticks=10,
    fs_legend=14,
    w_total=0.9,
):
    df = get_df()
    t_cols, _ = get_t_cols(p=df)
    docs = df.sort_values("date_of_collection").date_of_collection.unique()[-4:-1]
    mask_doc = np.array([doc in docs for doc in df.date_of_collection])
    mask_antibiotic = df.antibiotic_type != "none"
    df = df.loc[mask_doc & mask_antibiotic].copy()

    n_rows = n_panels // n_cols
    x_step = 1 / n_cols * w_total
    y_step = 1 / n_rows

    for i_ax, strain in enumerate(reversed(df.Strain.unique())):
        mask = df.Strain == strain
        i_col = n_cols - i_ax % n_cols - 1
        i_row = i_ax // n_cols
        a = ax.inset_axes(
            [i_col * x_step + p_x, i_row * y_step + p_y, x_step - 2 * p_x, y_step - 2 * p_y,]
        )
        a.text(
            10,
            y_max,
            (strain[:4] + strain[-2:]).replace("_", ""),
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
        )
        for i, (antibiotic, concentration) in enumerate([
            ("amoxicillin", 5.0,),
            ("carbenicillin", 30.0,),
            ("cefotaxime", 5.0,),
        ]):
            mask_condition = mask & (df.antibiotic_type == antibiotic) & (df.antibiotic_conc == concentration)
            if (i_col == (n_cols - 1)) and (i_row == (n_rows - 1)):
                a.plot(
                    df.loc[mask_condition, t_cols].to_numpy()[0],
                    color=colors[i],
                    lw=lw,
                    alpha=alpha,
                    label=antibiotic,
                )
                a.plot(
                    df.loc[mask_condition, t_cols].to_numpy()[1:].T,
                    color=colors[i],
                    lw=lw,
                    alpha=alpha,
                )
            else:
                a.plot(
                    df.loc[mask_condition, t_cols].to_numpy().T,
                    color=colors[i],
                    lw=lw,
                    alpha=alpha,
                )
        a.set_ylim(0, y_max)
        a.set_xlim(0, 128,)
        if (i_col == (n_cols - 1)) and (i_row == (n_rows - 1)):
            a.legend(
                bbox_to_anchor=(1.05, 1,),
                fontsize=fs_legend,
            )

        if (i_col == 0) and (i_row == 0):
            a.set_xticks(
                [0, 128,],
                labels=[0, 128,],
                fontsize=fs_ticks,
            )
            a.set_yticks(
                [0, y_max,],
                labels=[0, y_max,],
                fontsize=fs_ticks,
            )
            a.set_xlabel(
                "Time",
                fontsize=fs_label,
            )
            a.set_ylabel(
                "Cell density",
                fontsize=fs_label,
            )
            for spine in ["top", "right",]:
                a.spines[spine].set_visible(False)    
        else:
            a.axis("off")
        if i_ax == (n_panels - 1):
            break
