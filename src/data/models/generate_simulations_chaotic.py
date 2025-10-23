import numpy as np
from tqdm import trange

from config import DIR_DATA
from data.models.chaotic_model import ChaoticModel


def create_data(file_prefix):
    model = ChaoticModel(file_prefix=file_prefix, t_max=1000,)

    y0, alpha = model.load_cached_params()
    sol = model.solve_ode(y0=y0, alpha=alpha, t_eval=np.linspace(0, model.t_span[1], 128 * 30))

    y = None
    for chunk in range(sol.y.shape[1] // 128):
        idx = chunk * 128
        if y is None:
            y = sol.y[:, idx:idx+128]
        else:
            y = np.concatenate((
                y,
                sol.y[:, idx:idx+128],
            ))
    y = y[y.max(axis=1) > 0.0001]
    y /= y.max(axis=1).reshape(-1, 1)

    np.save(
        DIR_DATA
        / "simulation"
        / "chaotic"
        / f"{model.model_name}_{model.file_prefix}_y_raw.npy",
        sol.y,
    )


def make_y(filename):

    y = np.load(
        DIR_DATA
        / "simulation"
        / "chaotic"
        / filename,
    )
    length_intermediate = y.shape[1]
    length_input = 128
    tgt_samples = 10000
    n_cols = tgt_samples // y.shape[0]
    overlap = int((y.shape[1] - n_cols * length_input) / (-n_cols + 1))
    n_cols = None
    if n_cols is None:
        n_cols = (length_intermediate - length_input) // (length_input - overlap) + 1
    
    y_new = np.zeros((
        y.shape[0] * n_cols,
        length_input,
    ))
    
    for i_row in trange(y.shape[0]):
        for i_col in range(n_cols):
            i1 = i_col * (length_input - overlap)
            i2 = i1 + length_input
            sample = y[i_row, i1:i2]
            if (sample.max() == sample.min()) or ((sample != 0).sum() == 1):
                continue
            sample = sample / sample.max()
            y_new[i_row * n_cols + i_col, :] = sample
    np.save(
        DIR_DATA
        / "simulation"
        / "chaotic"
        / filename.replace("_raw", ""),
        y_new,
    )


if __name__ == "__main__":
    for file_prefix in [
        "2024-06-03",
        "2024-06-03-v0",
        "2024-06-03-v1",
        "2024-06-03-v2",
        "2024-06-03-v3",
    ]:
        # create_data(file_prefix=file_prefix)
        # make_y(filename=f"chaotic_{file_prefix}_y_raw.npy")
        pass