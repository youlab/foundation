"""
Generate the data splits for the simulation-fraction sensitivity analysis.

The experimental corpus and its train/test partition are held EXACTLY fixed across all six
splits (they are the original experimental-only split stored in old_split); only the
simulation corpus is resampled and re-split. The simulation subsamples are nested
(10% subset of 20% subset of 40% subset of 80% subset of 100%), so differences along the
sensitivity curve reflect corpus size rather than resampling noise.
"""
import json

import numpy as np

from config import DIR_DATA_PROCESSED, SEQ_LEN
from data.config import RANDOM_SEED
from data.normalization_functions.utils import generate_train_test_idx

# the fixed experimental split and the full simulation corpus both live here
DIR_OLD_SPLIT = DIR_DATA_PROCESSED / "dataset_level_splitting_analysis" / "old_split"
DIR_OUT = DIR_DATA_PROCESSED

PERC_SIM = [0, 10, 20, 40, 80, 100]

N_EXP = 306417
N_SIM = 69674


def row_filename(entry):
    """Return the source filename stored in a per-row idx_key entry."""
    for key in entry:
        if key not in ["original_shape", "y.shape"]:
            return key
    return None


def main():
    # load the fixed experimental split
    exp = np.load(DIR_OLD_SPLIT / f"{SEQ_LEN}_2024-08-16_experimental.npz")
    y_exp = exp["y"]                    # (N_EXP, SEQ_LEN)
    exp_train_idx = exp["train_idx"]    # (245133,), order preserved verbatim
    exp_test_idx = exp["test_idx"]      # (61284,)

    # load the full simulation corpus
    sim = np.load(DIR_OLD_SPLIT / f"{SEQ_LEN}_2024-08-16_simulation.npz")
    y_sim = sim["y"] # (N_SIM, SEQ_LEN)

    with open(DIR_OLD_SPLIT / f"{SEQ_LEN}_2024-08-16_experimental_idx_key.json", "r") as fp:
        idx_key_exp = json.load(fp)
    with open(DIR_OLD_SPLIT / f"{SEQ_LEN}_2024-08-16_simulation_idx_key.json", "r") as fp:
        idx_key_sim = json.load(fp)

    assert y_exp.shape == (N_EXP, SEQ_LEN), f"y_exp shape {y_exp.shape}"
    assert y_sim.shape == (N_SIM, SEQ_LEN), f"y_sim shape {y_sim.shape}"
    assert y_exp.dtype == np.float64 and y_sim.dtype == np.float64
    assert np.array_equal(
        np.sort(np.concatenate((exp_train_idx, exp_test_idx))),
        np.arange(N_EXP),
    ), "experimental train/test indices are not a partition of the experimental corpus"

    # single permutation drawn once, outside the loop, so subsamples are nested
    perm = np.random.default_rng(RANDOM_SEED).permutation(N_SIM)

    for perc in PERC_SIM:
        print(f"\nWorking on perc_sim {perc}" + 40 * "=")

        # sorting keeps each source file's retained rows contiguous in the new corpus
        sel = np.sort(perm[:int(round(perc / 100 * N_SIM))])
        n_sel = int(sel.shape[0])

        if n_sel > 0:
            sim_train_pos, sim_test_pos = generate_train_test_idx(
                n=n_sel,
                use_random_seed=True,
            )
        else:
            sim_train_pos = np.zeros(0, dtype=int)
            sim_test_pos = np.zeros(0, dtype=int)

        y = np.concatenate((y_exp, y_sim[sel]))
        train_idx = np.concatenate((exp_train_idx, N_EXP + sim_train_pos)).astype(np.int64)
        test_idx = np.concatenate((exp_test_idx, N_EXP + sim_test_pos)).astype(np.int64)

        assert y.shape == (N_EXP + n_sel, SEQ_LEN), f"y shape {y.shape}"
        assert y.dtype == np.float64
        assert np.array_equal(y[:N_EXP], y_exp), "experimental block was not preserved"
        assert np.array_equal(
            np.sort(np.concatenate((train_idx, test_idx))),
            np.arange(y.shape[0]),
        ), "train/test indices are not a partition of the corpus"

        # the 100% corpus must reproduce the original combined corpus exactly
        if perc == 100:
            y_all_old = np.load(DIR_OLD_SPLIT / f"{SEQ_LEN}_2024-08-16_all.npz")["y"]
            assert np.array_equal(y, y_all_old), "100% corpus differs from the original all corpus"

        # build the idx_key; experimental part is unchanged, simulation part is remapped
        idx_key = dict(idx_key_exp)

        sim_files = {}
        for k in range(n_sel):
            entry = idx_key_sim[str(int(sel[k]))]
            idx_key[str(N_EXP + k)] = entry
            fn = row_filename(entry)
            if fn not in sim_files:
                sim_files[fn] = {"y_all_i": N_EXP + k, "n_retained": 0}
            sim_files[fn]["n_retained"] += 1

        # NOTE: in these directories "y.shape" for a simulation file is the number of RETAINED
        # rows after subsampling, not the number of rows compiled from that file
        idx_key["simulation"] = {}
        for fn in idx_key_sim["simulation"]:
            n_retained = sim_files.get(fn, {}).get("n_retained", 0)
            idx_key["simulation"][fn] = {
                "original_shape": idx_key_sim["simulation"][fn]["original_shape"],
                "y.shape": [n_retained, SEQ_LEN],
                "y_all_i": sim_files.get(fn, {}).get("y_all_i", N_EXP),
            }

        train_pct = len(train_idx) / y.shape[0] * 100
        test_pct = len(test_idx) / y.shape[0] * 100
        print(f"Train shape: {tuple(y[train_idx].shape)} ({train_pct:.3f}%)")
        print(f"Test shape: {tuple(y[test_idx].shape)} ({test_pct:.3f}%)")
        print(f"Simulation curves: {n_sel} of {N_SIM}")

        run_dir = DIR_OUT / f"perc_sim_{perc:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            run_dir / f"{SEQ_LEN}_all_y.npz",
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        with open(run_dir / f"{SEQ_LEN}_all_idx_key.json", "w") as fp:
            json.dump(idx_key, fp)

        data_config = {
            "perc_sim": perc,
            "random_seed": int(RANDOM_SEED),
            "split_source": "dataset_level_splitting_analysis/old_split",
            "n_experimental": N_EXP,
            "n_simulation": n_sel,
            "train_shape": list(y[train_idx].shape),
            "test_shape": list(y[test_idx].shape),
            "train_percentage": train_pct,
            "test_percentage": test_pct,
            # simulation fraction of the training set
            "sim_fraction_of_train": len(sim_train_pos) / len(train_idx),
        }

        with open(run_dir / "data_config_all.json", "w") as fp:
            json.dump(data_config, fp, indent=2)

        print(f"Saved to {run_dir}")
