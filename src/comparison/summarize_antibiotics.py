import pandas as pd

from ml.utils.load_models import get_best_models
from config import (
    DIR_RESULTS_ANTIBIOTICS,
    DIR_RESULTS_MODEL_COMPARISON,
)


def main():
    models = get_best_models()

    for prefix in [
        "classify_antibiotic_summary",
        "classify_antibiotics_CIP_summary",
        "classify_antibiotics_GM_summary",
        "classify_antibiotics_SAM_summary",
        "classify_antibiotics_SXT_summary",
        "regress_antibiotic_summary",
    ]:
        df = None
        for model_type, z_dim in models:
            try:
                temp = pd.read_csv(
                    DIR_RESULTS_ANTIBIOTICS
                    / f"{prefix}_{model_type}_{z_dim}.csv",
                    index_col=0,
                )
                temp["model_name"] = f"{model_type}_{z_dim}"
                temp["model_type"] = model_type
                temp["z_dim"] = z_dim
        
                df = pd.concat(
                    (
                        df,
                        temp,
                    )
                )
            except Exception as e:
                print(f"Error with {prefix} {model_type} {z_dim}: {e}")
        df = df.loc[df.train_size_pct == 100, :].copy()

        df_out = None
        for col in ["raw_accuracy", "latent_accuracy",]:
            temp = df.loc[:, ["model_type", "z_dim", col]].copy()
            temp = temp.rename(columns={col: "accuracy"})
            if col.find("raw") > -1:
                temp["input_type"] = "raw"
            else:
                temp["input_type"] = "latent"
            df_out = pd.concat(
                (
                    df_out,
                    temp,
                )
            )

        df_out = df_out.pivot(
            columns="z_dim",
            index=["model_type", "input_type",],
            values=["accuracy",],
        )
        df_out = df_out.loc[:, "accuracy"]
        df_out.round(3).to_csv(
            DIR_RESULTS_MODEL_COMPARISON
            / f"{prefix}_pivot.csv",
        )
