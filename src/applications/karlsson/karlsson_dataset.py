import logging
import pickle
from pathlib import Path
from datetime import datetime
import re
import os
import json

import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.decomposition import PCA

from ml.utils.load_models import load_default_model
from applications.utils.latents import get_latents, decode
from applications.karlsson.utils import (
    parse_omnilog_csv,
)

from config import (
    SEQ_LEN,
    Z_DIM,
)


def preprocess_data(
    data_dir, 
    output_path: str = None,
    output_type: str = 'dataframe' # 'dataframe' or 'json'
):
    '''load karlsson data'''
    files = sorted(Path(data_dir).glob('SLF_*kinetic.csv'))
    all_data = []

    # run through files to load data
    for f in files:
        print(f'Parsing raw file {f.name}')
        file_data = parse_omnilog_csv(f)
        all_data.extend(file_data)

    # output and save based on output_type
    if output_type == 'dataframe':
        data = pd.DataFrame(all_data)
        if output_path:
            data.to_csv(output_path, index=False)
    elif output_type == 'json':
        data = all_data.copy()
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(data, f)
    
    return data


def train_test_split_dark(
    data,
    train_size: float = 0.8, 
    split_level: str = 'per_replicate' # 'per_replicate' or 'per_species'
):
    '''create train test split for dark data only and return indices'''

    # convert to pd.DataFrame type
    if not isinstance(data, pd.DataFrame):
        data = pd.read_json(data.copy())

    # get grouping to stratify on
    dark_data = data[data['treatment'] == 'dark']
    if split_level == "per_replicate":
        grouped = dark_data.groupby(
            ["organism", "replicate"],
            sort=True,
        )
    elif split_level == "per_species":
        grouped = dark_data.groupby(
            ["organism"],
            sort=True,
        )
    else:
        raise ValueError(f"Unknown split_level '{split_level}', expected 'per_replicate' or 'per_species'.")

    # get indices
    group_indices = [
        group.index.tolist()
        for _, group in grouped
    ]

    n_groups = len(group_indices)
    n_test_groups = max(1, int(np.ceil((1 - train_size) * n_groups)))

    # perform stratified sampling
    test_groups = np.random.choice(
        n_groups,
        size=n_test_groups,
        replace=False,
    )

    train_indices = []
    test_indices = []

    # organize resulting indices
    for group, indices in enumerate(group_indices):
        if group in test_groups:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return sorted(train_indices), sorted(test_indices)


def downsample_train(
    n,
    full_train_indices,
    downsampling_seed,
    output_path: str = None,
):
    '''downsamples given train indices'''

    # return all indices if n > total number of indices
    if n >= len(full_train_indices):
        print(f'[WARNING] Specified downsampled train size of {n} is greater than or equal to the total number of train indices {len(full_train_indices)}')
        return full_train_indices

    # perform downsampling
    rng = np.random.default_rng(seed=downsampling_seed)
    downsampled_indices = rng.choice(
        full_train_indices,
        size=n,
        replace=False,
    )
    downsampled_indices = np.sort(downsampled_indices).tolist()

    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'downsampling_seed': int(downsampling_seed),
                'len_downsampled_train_indices': len(downsampled_indices),
                'prop_downsampled_train_indices': float(len(downsampled_indices)) / len(full_train_indices),
                'downsampled_train_indices': downsampled_indices,
            }, f, indent=2)

    return downsampled_indices


class KarlssonDarkDataLoader:
    '''prep data into windows ready for regression'''

    def __init__(
        self,
        data,
        train_indices,
        test_indices,
        input_type='raw',
        target_type='raw',
        window_size=SEQ_LEN,
        stride=1,
        encoder=None,
        pca_components=Z_DIM,
        append_max=True,
        random_seed=501,
    ):
        self.data = data
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.input_type = input_type
        self.target_type = target_type
        self.window_size = window_size
        self.stride = stride
        self.pca_components = pca_components
        self.append_max = append_max
        self.random_seed = random_seed

        # only pay for the encoder when a latent representation is actually requested
        if (input_type == 'latent') or (target_type == 'latent'):
            self.encoder = encoder if encoder is not None else load_default_model()
        else:
            self.encoder = encoder

        # store PCA encoders
        self.input_pca = None
        self.target_pca = None
        self.shared_pca = None


    def get_regression_data(self):
        '''
        Returns
        -------
        X_train, y_train, X_test, y_test
            Arrays ready for ExtraTreesRegressor
        '''

        train_traj = self._get_trajectories(self.train_indices)
        test_traj = self._get_trajectories(self.test_indices)

        X_train_raw, y_train_raw = self._extract_window_pairs(train_traj)
        X_test_raw, y_test_raw = self._extract_window_pairs(test_traj)

        self._fit_pca(
            X_train_raw,
            y_train_raw,
        )

        X_train = self._transform_windows(
            X_train_raw,
            representation=self.input_type,
            side="input",
        )

        X_test = self._transform_windows(
            X_test_raw,
            representation=self.input_type,
            side="input",
        )

        y_train = self._transform_windows(
            y_train_raw,
            representation=self.target_type,
            side="target",
        )

        y_test = self._transform_windows(
            y_test_raw,
            representation=self.target_type,
            side="target",
        )

        return X_train, y_train, X_test, y_test


    def _get_trajectories(self, indices):
        '''extract trajectories from dataframe'''
        trajectories = []
        for traj in self.data.loc[indices, 'trajectory']:
            trajectories.append(
                np.asarray(traj[:384], dtype=np.float32) # use first 384 time points only, discard last
            )

        return trajectories


    def _extract_window_pairs(self, trajectories):
        '''extract window input-target pairs'''
        X = []
        y = []
        w = self.window_size

        for trajectory in trajectories:

            n = len(trajectory)
            max_input_start = n - (2 * w) # leave enough room for last input window to have a full target window

            for start in range(0, max_input_start + 1, self.stride):
                input_window = trajectory[start : start + w]
                target_window = trajectory[start + w : start + (2 * w)]

                X.append(input_window)
                y.append(target_window)

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        )


    def _normalize_curves(self, curves):
        '''divide each curve by its own maximum, returning the normalised curves and the maxima'''

        # curves has shape (n_windows, window_size)
        maxima = curves.max(axis=1, keepdims=True)

        # zero curves are mapped to ones exactly like get_latents does
        normalized = np.ones_like(curves)
        np.divide(
            curves,
            maxima,
            out=normalized,
            where=maxima > 0,
        )

        return normalized, maxima


    def _fit_pca(self, X_train_raw, y_train_raw,):
        '''fit PCA functions with training data'''

        needs_input_pca = (self.input_type == "pca")
        needs_target_pca = (self.target_type == "pca")

        if not (needs_input_pca or needs_target_pca):
            return

        # PCA fit on max normalized curves to match A7X
        X_curves, _ = self._normalize_curves(X_train_raw)
        y_curves, _ = self._normalize_curves(y_train_raw)

        # shared PCA space
        if (self.input_type == "pca" and self.target_type == "pca"):
            all_windows = np.concatenate(
                [X_curves, y_curves],
                axis=0,
            )
            self.shared_pca = PCA(
                n_components=self.pca_components
            )
            self.shared_pca.fit(all_windows)

            return

        # input-only PCA
        if needs_input_pca:
            self.input_pca = PCA(
                n_components=self.pca_components
            )
            self.input_pca.fit(X_curves)

        # target-only PCA
        if needs_target_pca:
            self.target_pca = PCA(
                n_components=self.pca_components
            )
            self.target_pca.fit(y_curves)


    def _transform_windows(
        self,
        windows,
        representation,
        side,
    ):
        '''returns windows in specified formats'''
        if representation == "raw":
            return windows.astype(np.float32)

        if representation == "latent":
            return self._encode_latent(
                windows
            )

        if representation == "pca":
            return self._encode_pca(
                windows,
                side,
            )

        raise ValueError(
            f'Unknown representation type: {representation}'
        )


    def _encode_latent(self, windows):
        '''latent encoder wrapper, get_latents normalises each curve and returns its max last'''

        # get_latents returns (n_windows, Z_DIM + 1), the final column is the appended max value
        features = get_latents(
            z=windows,
            model=self.encoder,
            batch_size=1024,
        )

        # drop max if append_max is not on
        if not self.append_max:
            features = features[:, :-1]

        assert features.shape == (windows.shape[0], Z_DIM + int(self.append_max))
        return features.astype(np.float32)


    def _encode_pca(self, windows, side):
        '''PCA encoder wrapper'''

        pca_input, maxima = self._normalize_curves(windows)

        if self.shared_pca is not None:
            components = self.shared_pca.transform(pca_input)
        elif side == "input":
            components = self.input_pca.transform(pca_input)
        elif side == "target":
            components = self.target_pca.transform(pca_input)
        else:
            raise ValueError(f"Unknown side, not input or target: {side}")

        if self.append_max:
            components = np.concatenate(
                [components, maxima],
                axis=1,
            )

        assert components.shape == (windows.shape[0], self.pca_components + int(self.append_max))
        return components.astype(np.float32)