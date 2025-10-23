import pickle
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    DIR_CACHE_CONSORTIA,
    DIR_RESULTS_CONSORTIA,
)


def get_data():
    with open(
        DIR_CACHE_CONSORTIA
        / "data_delta.pkl",
        "rb",
    ) as fp:
        data = pickle.load(
            file=fp,
        )
    return data


def main():
    data = get_data()
    dfrp = pd.DataFrame.from_records(data["regr_pred"])
    dffo = pd.DataFrame.from_records(data["future_outlook"])

    for consortium in dfrp.consortia.unique():
        fig = plt.figure(
            figsize=(16, 5,),
            layout="tight",
        )
        
        ax_dict = fig.subplot_mosaic(
            "ABE;CDF",
            width_ratios=[1,5,5],
        )
        
        ax_label_regr = ["A", "C",]
        ax_label_fc = ["B", "D",]
        ax_label_rmse = ["E", "F",]
        
        colors = ["firebrick", "cornflowerblue", "mediumorchid", "seagreen", "goldenrod", "silver",]
        lses = [":", "-",]
        for i_row, train_size in enumerate(["200", "8000",]):
            for i_tt, tgt_type in enumerate(["segment128", "latent", "segment16", "sliders16", "sliders32", "sliders64",]):
                for i_it, input_type in enumerate(["raw", "latent",]):
                    if (input_type == "raw") and (tgt_type == "latent"):
                        continue
                    if (input_type == "latent") and (tgt_type == "segment128"):
                        continue
                    mask = (dfrp.consortia == consortium) & (dfrp.tgt_type == tgt_type) & (dfrp.input_type == input_type) & (dfrp.train_size == train_size)
                    if mask.sum() == 0:
                        continue
                    ax_dict[ax_label_regr[i_row]].scatter(
                        i_it,
                        dfrp.loc[mask, "r2_test"],
                        color=colors[i_tt],
                        label=f"{input_type} {tgt_type}",
                    )
                    mask = (dffo.consortia == consortium) & (dffo.tgt_type == tgt_type) & (dffo.input_type == input_type) & (dffo.train_size == train_size)
                    if mask.sum() == 0:
                        continue
                    ax_dict[ax_label_fc[i_row]].plot(
                        dffo.loc[mask, "r2_test"].iloc[0],
                        color=colors[i_tt],
                        ls=lses[i_it],
                        label=f"{input_type} {tgt_type}",
                    )
                    ax_dict[ax_label_rmse[i_row]].plot(
                        dffo.loc[mask, "rmse_test"].iloc[0],
                        color=colors[i_tt],
                        ls=lses[i_it],
                        label=f"{input_type} {tgt_type}",
                    )
        
            if i_row == 1:
                ax_dict[ax_label_regr[i_row]].set_xticks(
                    [0, 1,],
                    labels=["raw", "latent",],
                    fontsize=14,
                )
            else:
                ax_dict[ax_label_regr[i_row]].set_xticks(
                    [0, 1,],
                    labels=[],
                )
            ax_dict[ax_label_regr[i_row]].set_xlim(-0.5, 1.5,)
        
            if i_row == 0:
                ax_dict[ax_label_rmse[i_row]].legend(
                    bbox_to_anchor=(1.05, 1.0,),
                )
                ax_dict[ax_label_regr[i_row]].set_ylabel(r"Regression model $R^2$")
                ax_dict[ax_label_fc[i_row]].set_ylabel(r"$R^2$")
                ax_dict[ax_label_rmse[i_row]].set_ylabel("RMSE")


            ax_dict[ax_label_regr[i_row]].set_ylim(-0.1, 1.1)
            ax_dict[ax_label_fc[i_row]].set_ylim(-0.1, 1.1)
            ax_dict[ax_label_rmse[i_row]].set_ylim(-0.05, 0.3)
        for ax_label in ax_dict.keys():
            for spine in ["top", "right",]:
                ax_dict[ax_label].spines[spine].set_visible(False)
        
        plt.savefig(
            DIR_RESULTS_CONSORTIA
            / f"segments_{consortium}.png",
        )
        plt.close()
