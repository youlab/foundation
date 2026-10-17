import os

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    DIR_FIGS_MANUSCRIPT,
    DIR_RESULTS,
)


# cached summary of the focal/background sweep on the original manuscript simulations (multi target mode)
SUMMARY_PATH = DIR_RESULTS / "consortia_focal_background" / "focal_background_FINAL" / "summary_multi_target.csv"

# (simulation, title, observed counts swept, x ticks, x limits, R2 y limits, RMSE y limits, RMSE y ticks)
COMMUNITIES = [
    ("bgLV", "Simple community", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], (0.5, 5.5), (0.75, 1.01), (0, 0.01), [0, 0.005, 0.01]),
    ("dgLV", "Complex community", [1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 6, 8], (0.5, 8.5), (0.75, 1.01), (0.04, 0.09), [0.04, 0.06, 0.08]),
]

REPRESENTATIONS = [
    "latent_to_raw",
    "raw_to_raw",
]

# the latent run appended the segment maximum to the A7X latents (append_max=True)
REPRESENTATION_LABELS = {
    "latent_to_raw": "A7X + Max to Raw",
    "raw_to_raw": "Raw to Raw",
}

REPRESENTATION_COLORS = {
    "latent_to_raw": "tab:blue",
    "raw_to_raw": "tab:orange",
}

N_REPEATS = 5


def main(
    fs_panel,
    fw_panel,
):
    '''
    Supplementary Figure S14 (rebuttal Figure R4): global test R2 and RMSE against number of observed focal taxa for the
    simple (bgLV) and complex (dgLV) simulated communities, mean +/- std over repeated focal subsets and train/test splits.
    '''
    FS_LABEL = 24
    FS_TICKS = 22
    FS_LEGEND = 20
    FS_TITLE = 30
    FS_PANEL_LETTER = fs_panel + 6  # larger than the shared panel letter size to match the larger fonts here

    df = pd.read_csv(SUMMARY_PATH)
    assert (df.target_mode == "multi_target").all(), "expected multi_target summary"
    assert (df.n_repeats == N_REPEATS).all(), f"expected {N_REPEATS} repeats for every row"
    assert df.loc[df.representation == "latent_to_raw", "append_max"].astype(bool).all(), "expected append_max=True for latent rows"

    # ax: (4,), a = simple R2, b = simple RMSE, c = complex R2, d = complex RMSE
    fig, ax = plt.subplots(1, 4, figsize=(24, 7))

    for i_community, (simulation, title, observed_counts, xticks, xlim, ylim_r2, ylim_rmse, yticks_rmse) in enumerate(COMMUNITIES):
        a_r2 = ax[2 * i_community]
        a_rmse = ax[2 * i_community + 1]

        for representation in REPRESENTATIONS:
            tmp = df[(df.simulation == simulation) & (df.representation == representation)].sort_values("n_observed")
            assert tmp.n_observed.tolist() == observed_counts, f"{simulation} {representation}: unexpected observed counts {tmp.n_observed.tolist()}"

            for a, metric in [(a_r2, "global_r2_test"), (a_rmse, "global_rmse_test")]:
                a.errorbar(
                    tmp.n_observed,
                    tmp[f"{metric}_mean"],
                    yerr=tmp[f"{metric}_std"],
                    marker="o",
                    markersize=8,
                    capsize=4,
                    linewidth=1.5,
                    color=REPRESENTATION_COLORS[representation],
                    label=REPRESENTATION_LABELS[representation],
                )

            last = tmp.iloc[-1]
            print(
                f"{title} ({simulation}), {REPRESENTATION_LABELS[representation]}, m = {last.n_observed}: "
                f"global R2 = {last.global_r2_test_mean:.4f}, global RMSE = {last.global_rmse_test_mean:.4f}"
            )

        for a in [a_r2, a_rmse]:
            a.set_xticks(xticks)
            a.set_xlim(xlim)
            a.set_xlabel("Number of observed focal taxa", fontsize=FS_LABEL)

        a_r2.set_ylim(ylim_r2)
        a_r2.set_yticks([0.8, 0.9, 1.0])
        a_r2.set_ylabel("Global R²", fontsize=FS_LABEL)

        a_rmse.set_ylim(ylim_rmse)
        a_rmse.set_yticks(yticks_rmse)
        a_rmse.set_ylabel("Global RMSE", fontsize=FS_LABEL)
        a_rmse.legend(loc="upper right", fontsize=FS_LEGEND)


    for i_panel, a in enumerate(ax):
        a.set_box_aspect(1)
        a.tick_params(labelsize=FS_TICKS)
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)
        a.text(
            -0.3,
            1.1,
            "abcd"[i_panel],
            transform=a.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=FS_PANEL_LETTER,
            fontweight=fw_panel,
        )

    # explicit margins so all labels lie inside the canvas (no bbox_inches="tight")
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.78, wspace=0.5)

    # community titles centered above their two panels; apply_aspect resolves the square panel positions before reading them
    for a in ax:
        a.apply_aspect()
    for i_community, (_, title, *_) in enumerate(COMMUNITIES):
        pos_r2 = ax[2 * i_community].get_position()
        pos_rmse = ax[2 * i_community + 1].get_position()
        fig.text(
            (pos_r2.x0 + pos_rmse.x1) / 2,
            pos_r2.y1 + 0.08,
            title,
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=FS_TITLE,
        )

    os.makedirs(DIR_FIGS_MANUSCRIPT, exist_ok=True)
    plt.savefig(DIR_FIGS_MANUSCRIPT / "supp_fig_14_new.png", dpi=300)
    plt.savefig(DIR_FIGS_MANUSCRIPT / "supp_fig_14_new.pdf", dpi=300)
    plt.savefig(DIR_FIGS_MANUSCRIPT / "supp_fig_14_new.svg", dpi=300)
    plt.close()
    print(f"Saved figure to {DIR_FIGS_MANUSCRIPT / 'supp_fig_14_new.pdf'}")
