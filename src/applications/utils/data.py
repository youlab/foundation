import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import GroupKFold, train_test_split

from config import SEQ_LEN


def get_t_cols(p):
    cols = []
    t = []
    for c in p:
        try:
            t.append(float(c))
            cols.append(c)
        except:
            try:
                t.append(float(c.split("before_ab_")[1]))
                cols.append(c)
            except:
                pass
    return cols, np.array(t)


def interpolate_y(y):
    f = interp1d(
        x=np.linspace(0, 1, y.shape[1],),
        y=y,
        assume_sorted=True,
    )
    return f(np.linspace(0, 1, SEQ_LEN,))


def cross_val_data_split(
    x,
    n_cross_val,
    k_fold,
):
    test_size = x.shape[0] // k_fold
    x_train, x_test = train_test_split(
        x,
        test_size=test_size,
        random_state=42,
    )

    if n_cross_val == 0:
        return x_train, x_test
    
    processed = []
    for i in range(n_cross_val,):
        if i == (k_fold - 2):
            x_train, x_test = x_test, x_train
        else:
            processed.append(x_test)
            x_train, x_test = train_test_split(
                x_train,
                test_size=test_size,
                random_state=42,
            )
    processed.append(x_train)
    x_train = np.concatenate(processed)
    return x_train, x_test


def group_cross_val_data_split(x, groups, n_cross_val, k_fold=5):
    """Deterministic GroupKFold: all curves from an isolate share a fold."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(x):
        raise ValueError("groups must be one-dimensional and aligned with x")
    if not 0 <= n_cross_val < k_fold:
        raise ValueError("n_cross_val must be between 0 and k_fold - 1")
    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=k_fold).split(x, groups=groups)
    ):
        if fold == n_cross_val:
            return x[train_idx], x[test_idx]
