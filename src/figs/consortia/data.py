import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from applications.config import NOW_TEXT
from applications.utils.metrics import get_rmse
from config import DIR_CACHE_CONSORTIA, MODEL_TYPE, SEQ_LEN, Z_DIM


def load_data(consortia, train_size, cross_val=0):
    """Build Figure 4's examples and metrics from the selected forecast caches.

    Each row describes one actual fold, without padding missing folds with zeros.
    The initial observed window is excluded from forecast accuracy metrics.
    """
    rows = []
    for consortium in consortia:
        reference_true = None
        for input_type, target_type in (("latent", "latent"), ("raw", "segment128")):
            path = DIR_CACHE_CONSORTIA / (
                f"future_outlook_{input_type}_{consortium}_{target_type}_{MODEL_TYPE}_{Z_DIM}_"
                f"{cross_val}_{train_size}_{NOW_TEXT}.npz"
            )
            with np.load(path) as data:
                true, predicted = data["w_test"], data["w_test_pred"]
            if true.shape != predicted.shape or true.ndim != 3 or true.shape[-1] <= SEQ_LEN:
                raise ValueError(f"Invalid forecast shapes in {path}: {true.shape}, {predicted.shape}")
            if not (np.isfinite(true).all() and np.isfinite(predicted).all()):
                raise ValueError(f"Non-finite forecasts in {path}")
            if reference_true is None:
                reference_true = true
            elif not np.array_equal(true, reference_true):
                raise ValueError(f"Raw and latent test communities do not match for {consortium}")
            n_time = true.shape[-1]
            r2 = r2_score(
                true.reshape(-1, n_time)[:, SEQ_LEN:],
                predicted.reshape(-1, n_time)[:, SEQ_LEN:],
                multioutput="raw_values",
            )
            _, rmse, _ = get_rmse(y_true=true, y_pred=predicted)
            rows.append({
                "consortium": consortium,
                "train_size": train_size,
                "input_type": input_type,
                "cross_val": cross_val,
                "n_folds": 1,
                "r2_test": r2,
                "rmse_test": rmse,
                "test_example": (true[0].copy(), predicted[0].copy()),
                "source_file": str(path),
            })
    return pd.DataFrame.from_records(rows)
