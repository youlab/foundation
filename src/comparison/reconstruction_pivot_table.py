import pandas as pd

from config import DIR_RESULTS_MODEL_COMPARISON


def main():
    df = pd.read_csv(
        DIR_RESULTS_MODEL_COMPARISON
        / "reconstruction.csv",
        index_col=0,
    )

    df_out = None
    for col in ["r2_train_all", "r2_train_sim", "r2_train_exp", "r2_test_all", "r2_test_sim", "r2_test_exp",]:
        temp = df.loc[:, ["model_name", "model_type", "z_dim", col]].copy()
        temp = temp.rename(columns={col: "r2"})
        if col.find("train") > -1:
            temp["method"] = "train"
        else:
            temp["method"] = "test"
        if col.find("all") > -1:
            temp["training_dataset"] = "both"
        elif col.find("exp") > -1:
            temp["training_dataset"] = "exp"
        else:
            temp["training_dataset"] = "sim"
        df_out = pd.concat(
            (
                df_out,
                temp,
            )
        )

    df_out.pivot(
        columns="z_dim",
        index=["model_type", "method", "training_dataset",],
        values=["r2",],
    ).round(3).to_csv(
        DIR_RESULTS_MODEL_COMPARISON
        / "reconstruction_pivot.csv",
    )
