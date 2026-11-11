"""Rebuild plotting inputs without rendering figures or rerunning unrelated analyses."""
import argparse
import json
import os
from pathlib import Path


def compile_inputs(output):
    import numpy as np
    from data.compile import compile_y
    from data.normalization_functions.utils import generate_train_test_idx

    output.mkdir(parents=True, exist_ok=True)
    for label, categories in [("experimental", ["experimental"]),
                              ("simulation", ["simulation"]),
                              ("all", ["experimental", "simulation"])]:
        print(f"Compiling {label}", flush=True)
        y, index = compile_y(seq_len=128, categories=categories, interp_len=128)
        train, test = generate_train_test_idx(n=y.shape[0], use_random_seed=True)
        np.savez(output / f"128_{label}_y.npz", y=y, train_idx=train, test_idx=test)
        (output / f"128_{label}_idx_key.json").write_text(json.dumps(index))
        print(label, y.shape, flush=True)


def forecast_inputs(kind, source, output):
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import train_test_split
    from applications.config import NOW_TEXT
    from applications.datasets.raw_dataset import RawDataset
    from applications.utils.latents import get_latents
    from applications.utils.data import cross_val_data_split
    from applications.consortia.future_outlook import get_predictions
    from ml.utils.load_models import load_default_model

    output.mkdir(parents=True, exist_ok=True)
    filename = {"simple": "bgLV_B15_T5_fixed.txt", "complex": "dgLV_B92_T8_fixed.txt"}[kind]
    raw = RawDataset(source / "src/applications/consortia/data_files" / filename,
                     interp_len=768, train_size=8000)
    model = load_default_model()
    # Encode one window at a time to avoid materializing all raw windows on the GPU.
    x, target = [], []
    for start in range(0, 768 - 256, 32):
        print(f"Encoding {kind} window {start}", flush=True)
        x.append(get_latents(raw.y_train[:, start:start + 128], model=model).reshape(-1, raw.n_focal * 9))
        target.append(get_latents(raw.y_train[:, start + 128:start + 256], model=model).reshape(-1, raw.n_focal * 9))
    x, target = np.concatenate(x), np.concatenate(target)
    x[np.isnan(x)] = 0
    target[np.isnan(target)] = 0
    x, _ = cross_val_data_split(x, n_cross_val=0, k_fold=5)
    target, _ = cross_val_data_split(target, n_cross_val=0, k_fold=5)
    w_test = raw.y_test.reshape(-1, raw.n_focal, 768)
    for size in (12800, 102400):
        if size == len(x):
            x_train, y_train = x, target
        else:
            x_train, _, y_train, _ = train_test_split(x, target, train_size=size, random_state=42)
        print(f"Training {kind}: {size} windows, depth 15", flush=True)
        regr = ExtraTreesRegressor(random_state=42, max_depth=15, n_jobs=2)
        regr.fit(x_train, y_train)
        prediction = get_predictions(w_test, raw.n_focal, regr, False, "latent", model, "A7X", 8)
        path = output / f"future_outlook_latent_{kind}_latent_A7X_8_0_{size}_{NOW_TEXT}.npz"
        np.savez(path, w_test=w_test, w_test_pred=prediction)
        print(f"Saved {path}", flush=True)


def raw_forecast_inputs(kind, source, output):
    """Rebuild the raw-to-raw 128-point baseline for Figure 4's training size."""
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import train_test_split
    from applications.config import NOW_TEXT
    from applications.datasets.raw_dataset import RawDataset
    from applications.utils.data import cross_val_data_split

    output.mkdir(parents=True, exist_ok=True)
    filename = {"simple": "bgLV_B15_T5_fixed.txt", "complex": "dgLV_B92_T8_fixed.txt"}[kind]
    raw = RawDataset(source / "src/applications/consortia/data_files" / filename,
                     interp_len=768, train_size=8000)
    communities = raw.y_train.reshape(-1, raw.n_focal, 768)
    starts = list(range(0, 768 - 256, 32))
    indices, _ = cross_val_data_split(np.arange(len(starts) * len(communities)), 0, 5)
    indices, _ = train_test_split(indices, train_size=12800, random_state=42)
    x = np.empty((len(indices), raw.n_focal * 128), dtype=np.float32)
    target = np.empty_like(x)
    for window, start in enumerate(starts):
        mask = indices // len(communities) == window
        selected = communities[indices[mask] % len(communities)]
        x[mask] = selected[:, :, start:start + 128].reshape(-1, raw.n_focal * 128)
        target[mask] = selected[:, :, start + 128:start + 256].reshape(-1, raw.n_focal * 128)
    print(f"Training raw {kind}: {x.shape}, depth 15", flush=True)
    regr = ExtraTreesRegressor(random_state=42, max_depth=15, n_jobs=4)
    regr.fit(x, target)
    truth = raw.y_test.reshape(-1, raw.n_focal, 768)
    prediction = np.zeros_like(truth)
    prediction[:, :, :128] = truth[:, :, :128]
    for start in range(128, 768, 128):
        print(f"Forecasting raw {kind}: {start}", flush=True)
        prior = prediction[:, :, start - 128:start].reshape(len(truth), -1)
        prediction[:, :, start:start + 128] = np.clip(
            regr.predict(prior).reshape(len(truth), raw.n_focal, 128), 0, 1)
    path = output / f"future_outlook_raw_{kind}_segment128_A7X_8_0_12800_{NOW_TEXT}.npz"
    np.savez(path, w_test=truth, w_test_pred=prediction)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["processed", "simple", "complex"])
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-type", choices=["latent", "raw"], default="latent")
    args = parser.parse_args()
    os.environ["FOUNDATION_HOME"] = str(args.source_root.resolve())
    if args.kind == "processed":
        compile_inputs(args.output_dir)
    elif args.input_type == "raw":
        raw_forecast_inputs(args.kind, args.source_root, args.output_dir)
    else:
        forecast_inputs(args.kind, args.source_root, args.output_dir)
