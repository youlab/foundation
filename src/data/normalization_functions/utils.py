import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tqdm import trange

from config import DIR_DATA


def generate_train_test_idx(
    n,
    train_ratio=0.8,
):
    if not isinstance(n, int):
        return TypeError(f"n should be int but is {type(n)}")
    
    N_SET = 1000
    n_loops = n // N_SET
    idx = np.arange(n)
    n_train = int(n * train_ratio)
    n_test = n - n_train
    n_train_loop = int(N_SET * train_ratio)
    n_test_loop = N_SET - n_train_loop
    idx_train = np.zeros(n_train, dtype=int,)
    idx_test = np.zeros(n_test, dtype=int,)
    if n_loops > 0:
        for random_state in range(n_loops):
            idx_train[
                random_state * n_train_loop:(random_state + 1) * n_train_loop
            ], idx_test[
                random_state * n_test_loop:(random_state + 1) * n_test_loop
            ] = train_test_split(
                idx[random_state * N_SET:(random_state + 1) * N_SET],
                random_state=random_state,
                train_size=train_ratio,
            )
        i_add = n_loops * N_SET
    else:
        random_state = -1
        i_add = 0

    if n > (n_loops * N_SET):
        n_train_loop = int((n % N_SET) * train_ratio)
        n_test_loop = (n % N_SET) - n_train_loop
    
        idx_train[-n_train_loop:], idx_test[-n_test_loop:] = train_test_split(
            idx[i_add:],
            random_state=random_state + 1,
            train_size=train_ratio,
        )
    
    return idx_train, idx_test


def generate_train_test_idx_by_dataset(
    idx_key,
    train_ratio=0.8,
):
    """
    Generate train-test split such that all samples from each dataset file
    are assigned to either train or test and never split between them
    
    Parameters
    ----------
    idx_key : dict
        The index key from compile_y containing sample-to-file mappings. 
        idx_key[i] for integer i contains the filename as a key mapping to indices of samples within the file (to Derek's understanding)
    train_ratio : float
        Ratio of files to assign to train set, canonically set to 0.8
    
    Returns
    -------
    idx_train : np.ndarray
        Indices of all samples in train set
    idx_test : np.ndarray
        Indices of all samples in test set
    """

    # parse through idx_key to find all files and use files_to_sample to map files to sample indices
    files_to_samples = {}
    for i in idx_key:
        if isinstance(i, int) or (isinstance(i, str) and i.isdigit()):
            fn = None
            for key in idx_key[i]:
                if key not in ["original_shape", "y.shape"]:
                    fn = key
                    break
            if fn is not None:
                if fn not in files_to_samples:
                    files_to_samples[fn] = []
                files_to_samples[fn].append(int(i) if isinstance(i, str) else i)
    
    # do the train-test split on list of files
    files = list(files_to_samples.keys())
    file_idx_train, file_idx_test = generate_train_test_idx(n=len(files), train_ratio=train_ratio)
    train_files = [files[i] for i in file_idx_train]
    test_files = [files[i] for i in file_idx_test]
    
    # collect sample indices for train and test files post-splitting
    idx_train = []
    for file in train_files:
        idx_train.extend(files_to_samples[file])
    idx_test = []
    for file in test_files:
        idx_test.extend(files_to_samples[file])
    
    return np.array(sorted(idx_train)), np.array(sorted(idx_test))


def get_t_cols(p):
    cols = []
    t = []
    for c in p:
        try:
            t.append(float(c))
            cols.append(c)
        except:
            pass
    return cols, np.array(t)


def gp_y(
    y,
    num_days,
    length_input,
    overlap,
    no_mask_no_norm=False,
    n_cols=None,
    ignore_bad_row=False,
):
    N_REPEATS = 5
    overlap *= num_days
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    n_pts = (length_input - num_days) // (num_days - 1) + 1
    length_intermediate = int(y.shape[1] + (n_pts - 1) * (y.shape[1] - 1))
    x = np.linspace(
        0,
        length_intermediate - 1,
        y.shape[1],
        dtype=int,
    ).reshape(-1, 1)
    x_fit = np.linspace(
        0,
        length_intermediate - 1,
        length_intermediate,
        dtype=int,
    ).reshape(-1, 1)

    if n_cols is None:
        n_cols = (length_intermediate - length_input) // (length_input - overlap) + 1

    y_new = np.zeros((
        y.shape[0] * n_cols,
        length_input,
    ))
    
    for i_row in trange(y.shape[0]):
        row_hold = np.zeros((N_REPEATS, x_fit.shape[0]))
        row = y[i_row, :].reshape(-1, 1)
        for i_repeat in range(N_REPEATS):
            gaussian_process.fit(x, row)
            row_hold[i_repeat, :] = gaussian_process.predict(x_fit).flatten()
    
        row_fit = np.median(a=row_hold, axis=0,)
        
        for i_col in range(n_cols):
            i1 = i_col * (length_input - overlap)
            i2 = i1 + length_input
            sample = row_fit[i1:i2]
            if not ignore_bad_row:
                if (sample.max() == sample.min()) or ((sample != 0).sum() == 1):
                    continue
            if not no_mask_no_norm:
                sample = sample / sample.max()
            y_new[i_row * n_cols + i_col, :] = sample

    if no_mask_no_norm:
        return y_new
    y_new = y_new[np.isnan(y_new).sum(axis=1) == 0]
    mask = ~(y_new.max(axis=1) == y_new.min(axis=1))
    y = y_new[mask]
    y = (y - y.min(axis=1).reshape(-1, 1)) / (y.max(axis=1).reshape(-1, 1) - y.min(axis=1).reshape(-1, 1))
    return y


def get_ys(
    y_raw,
    filename=None,
    data_type="",
    num_days=None,
    length_input=None,
    overlap=None,
):
    y_filtered = filter_y(
        y=y_raw,
        data_type=data_type,
    )
    if data_type == "microbiome":
        y_filtered = gp_y(
            y=y_filtered,
            num_days=num_days,
            length_input=length_input,
            overlap=overlap,
        )
    y_min = y_filtered.min(axis=1,)
    y_min[y_min > 0] = 0
    y_max = y_filtered.max(axis=1,)
    y = y_filtered / y_max.reshape(-1, 1)
    return y, y_min, y_max


def save_vals(category, label, t, y_raw, y, y_min, y_max,):
    for val, arr in (("t", t,), ("y_raw", y_raw,), ("y", y,), ("y_min", y_min,), ("y_max", y_max,),):
        if arr is None:
            continue
        np.save(
            file=DIR_DATA / category / "processed" / f"{label}_{val}.npy",
            arr=arr,
        )


def smooth_y(y, window):
    if window % 2 == 1:
        return np.convolve(y, np.ones(window)/window)[int(window/2)-1:-int(window/2) - 1]
    return np.convolve(y, np.ones(window)/window)[int(window/2)-1:-int(window/2)]


def calc_r2s(y, window=5,):
    if window >= (y.shape[1] / 2):
        print(f"Array too short, window: {window} array length: {y.shape[1]}")
        return None
    r2s = np.zeros(y.shape[0])
    for idx in np.arange(y.shape[0]):
        y_ = smooth_y(y[idx], window)
        y_max = y[idx].max()
        r2s[idx] = r2_score(y_[window:-window] / y_max, y[idx][window:-window] / y_max)
    return r2s


def interpolate_y(y, interp_len,):
    x = np.arange(interp_len).astype(float)
    xp = np.linspace(0, interp_len, y.shape[1]).astype(int).astype(float)
    for i in range(y.shape[0]):
        if i == 0:
            y_temp = np.interp(
                x=x,
                xp=xp,
                fp=y[i, :],
            ).reshape(1, -1)
        else:
            y_temp = np.concatenate((
                y_temp,
                np.interp(
                    x=x,
                    xp=xp,
                    fp=y[i, :],
                ).reshape(1, -1),
            ))
    return y_temp


def filter_y(
    y,
    data_type,
):
    if data_type == "simulation":
        return y
    
    INTERP_LEN = 256
    R2_LIMIT = 0.8
    y[y < 0] = 0
    y = y[y.max(axis=1) != y.min(axis=1)]

    if data_type == "microbiome":
        return y
    
    if y.shape[1] != INTERP_LEN:
        y_interp = interpolate_y(y=y, interp_len=INTERP_LEN,)
    
    r2s = calc_r2s(y=y_interp)

    return y[r2s >= R2_LIMIT]


def calc_deviation(y, window=3, d_lim=0.1,):
    y_min = y.min(axis=1).reshape(-1, 1)
    y_max = y.max(axis=1).reshape(-1, 1)
    z = (y - y_min) / (y_max - y_min)
    sig2 = pd.DataFrame((z[:, 1:] - z[:, -1:]).T)
    d = ((sig2.abs() / sig2.abs().rolling(window).mean()).rolling(window).std()).mean().to_numpy().reshape(-1, 1)
    return ((d > d_lim) | (y_max < 1.2 * y_min)).flatten()


def robust_scaling(
    x_train, 
    x_test,
):
    q1 = np.percentile(x_train, 25,)
    q2 = np.percentile(x_train, 50,)
    q3 = np.percentile(x_train, 75,)

    x_train = (x_train - q2) / (q3 - q1)
    x_test = (x_test - q2) / (q3 - q1)
    return x_train, x_test
