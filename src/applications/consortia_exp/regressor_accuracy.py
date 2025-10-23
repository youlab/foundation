import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from applications.consortia_exp.config import get_config
from config import DIR_RESULTS_CONSORTIA_EXP


def main():
    source_files = [
        "soil-a",
        "soil-b",
        "soil-c",
        "water-a",
        "water-b",
        "water-c",
    ]
    results = []
    for source_file in source_files:
        _, test_reps = get_config(source_file=source_file)
        for use_raw in [True, False]:
            input_type = "raw" if use_raw else "latent"
            for tgt_type in ["latent", "segment128",]:
                tgt_test = []
                tgt_test_pred = []
                for test_rep in test_reps:
                    name_suffix = f"{source_file}-{test_rep}_{'raw' if use_raw else 'latent'}_{tgt_type}"
                    regr_pred = np.load(
                        DIR_RESULTS_CONSORTIA_EXP
                        / f"regr_pred_{name_suffix}_2025-02-03T21:54.npz",
                    ) 
                    tgt_test.append(regr_pred["tgt_test"].flatten())
                    tgt_test_pred.append(regr_pred["tgt_test_pred"].flatten())
                tgt_test = np.concatenate(tgt_test).flatten()
                tgt_test_pred = np.concatenate(tgt_test_pred).flatten()
                r2_test = r2_score(
                    y_true=tgt_test,
                    y_pred=tgt_test_pred,
                )
                results.append(
                    {
                        "source_file": source_file,
                        "input_type": input_type,
                        "tgt_type": tgt_type,
                        "r2_test": r2_test,
                    }
                )
    pd.DataFrame.from_records(results).to_csv(
        DIR_RESULTS_CONSORTIA_EXP
        / "consortia_exp_regressor_accuracy.csv",
    )
