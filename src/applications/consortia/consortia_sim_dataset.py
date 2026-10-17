from pathlib import Path
import json

import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

from ml.utils.load_models import load_default_model
from applications.utils.latents import get_latents

from config import (
    SEQ_LEN,
    Z_DIM,
)


def load_simulation_dataset(file_path, interp_len, focal_indices=None, simulation_indices=None):
    '''load one simulated focal/background dataset and interpolate it onto interp_len time points'''

    # memory map first so the focal and simulation axes can be subset before anything is
    # materialised, the interpolation below is what dominates memory and it scales with both
    dataset = np.load(file_path, mmap_mode='r')

    if dataset.ndim != 3:
        raise ValueError(
            f'{Path(file_path).name} has shape {dataset.shape}, expected (n_sim, n_focal, n_time).'
        )

    n_sim_on_disk, n_focal_on_disk, n_time = dataset.shape

    if simulation_indices is not None:
        simulation_indices = np.asarray(simulation_indices, dtype=int)
        if (simulation_indices.min() < 0) or (simulation_indices.max() >= n_sim_on_disk):
            raise ValueError(
                f'simulation_indices out of range for {Path(file_path).name} '
                f'with {n_sim_on_disk} simulations.'
            )
        dataset = dataset[simulation_indices]

    if focal_indices is not None:
        focal_indices = np.asarray(focal_indices, dtype=int)
        if (focal_indices.min() < 0) or (focal_indices.max() >= n_focal_on_disk):
            raise ValueError(
                f'focal_indices out of range for {Path(file_path).name} '
                f'with {n_focal_on_disk} focal populations.'
            )
        dataset = dataset[:, focal_indices, :]

    # np.asarray drops the memory map once both axes have been narrowed
    dataset = np.asarray(dataset)
    n_sim, n_focal, n_time = dataset.shape

    if simulation_indices is not None:
        assert n_sim == len(simulation_indices)
    if focal_indices is not None:
        assert n_focal == len(focal_indices)

    # the interpolation grid is built from integer sample positions, so upsampling only
    if interp_len < n_time:
        raise ValueError(
            f'interp_len ({interp_len}) must be at least the number of simulated time points ({n_time}).'
        )

    # interpolate along time, matching the convention used by the legacy RawDataset
    x2 = np.arange(interp_len).astype(float)
    x1 = (
        np.linspace(
            0,
            interp_len - 1,
            n_time,
        )
        .astype(int)
        .astype(float)
    )

    # interp1d works along the last axis, so flatten the simulation and focal axes together
    f = interp1d(
        x1,
        dataset.reshape(-1, n_time),
    )

    interpolated = f(x2)

    # abundances are bounded, clip any interpolation overshoot back into range
    interpolated[interpolated > 1] = 1
    interpolated[interpolated < 0] = 0

    dataset = interpolated.reshape(n_sim, n_focal, interp_len).astype(np.float32)
    assert dataset.shape == (n_sim, n_focal, interp_len)

    return dataset


def train_test_split_simulations(
    n_sim,
    train_size: float = 0.8,
    split_seed: int = 501,
    output_path: str = None,
):
    '''split simulation indices into train and test, the split unit is one whole simulation'''

    rng = np.random.default_rng(seed=split_seed)
    permuted = rng.permutation(n_sim)

    n_train = int(round(train_size * n_sim))

    if (n_train <= 0) or (n_train >= n_sim):
        raise ValueError(
            f'train_size {train_size} leaves an empty train or test split for n_sim={n_sim}.'
        )

    train_indices = sorted(permuted[:n_train].tolist())
    test_indices = sorted(permuted[n_train:].tolist())

    # splitting happens before any windowing, so no simulation can appear in both splits
    assert len(set(train_indices) & set(test_indices)) == 0
    assert (len(train_indices) + len(test_indices)) == n_sim

    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'split_seed': int(split_seed),
                'n_sim': int(n_sim),
                'n_train': len(train_indices),
                'n_test': len(test_indices),
                'train': train_indices,
                'test': test_indices,
            }, f, indent=2)

    return train_indices, test_indices


class ConsortiaSimDataLoader:
    '''prep simulated focal/background data into windows ready for regression'''

    def __init__(
        self,
        dataset,
        train_indices,
        test_indices,
        input_indices,
        target_indices,
        input_type='raw',
        target_type='raw',
        window_size=SEQ_LEN,
        stride=SEQ_LEN,
        encoder=None,
        z_dim=Z_DIM,
        pca_components=Z_DIM,
        append_max=True,
        random_seed=501,
    ):
        self.dataset = dataset
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.input_type = input_type
        self.target_type = target_type
        self.window_size = window_size
        self.stride = stride
        self.z_dim = z_dim
        self.pca_components = pca_components
        self.append_max = append_max
        self.random_seed = random_seed

        # which focal populations the regressor observes, and which it has to predict, these are
        # positions along axis 1 of dataset and are allowed to differ from each other
        self.input_indices = np.asarray(input_indices, dtype=int)
        self.target_indices = np.asarray(target_indices, dtype=int)

        n_focal_available = dataset.shape[1]

        if (self.input_indices.max() >= n_focal_available) or (self.target_indices.max() >= n_focal_available):
            raise ValueError(
                f'input_indices/target_indices exceed the {n_focal_available} focal populations present.'
            )

        # the observed focal populations of one simulation stay stacked into a single regression example
        self.n_input_focal = len(self.input_indices)
        self.n_target_focal = len(self.target_indices)

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

        X_train_raw, y_train_raw = self._extract_window_pairs(
            self.dataset[self.train_indices]
        )

        X_test_raw, y_test_raw = self._extract_window_pairs(
            self.dataset[self.test_indices]
        )

        self._fit_pca(
            X_train_raw,
            y_train_raw,
        )

        X_train = self._transform_windows(
            X_train_raw,
            representation=self.input_type,
            side='input',
        )

        X_test = self._transform_windows(
            X_test_raw,
            representation=self.input_type,
            side='input',
        )

        y_train = self._transform_windows(
            y_train_raw,
            representation=self.target_type,
            side='target',
        )

        y_test = self._transform_windows(
            y_test_raw,
            representation=self.target_type,
            side='target',
        )

        return X_train, y_train, X_test, y_test


    def _extract_window_pairs(self, data):
        '''extract window input-target pairs, vectorised over simulations and focal populations'''
        # data has shape (n_simulations, n_focal, interp_len)
        w = self.window_size
        n_time = data.shape[2]

        max_input_start = n_time - (2 * w) # leave enough room for last input window to have a full target window

        if max_input_start < 0:
            raise ValueError(
                f'Interpolated length {n_time} is too short for an input and target window of size {w}.'
            )

        starts = range(0, max_input_start + 1, self.stride)

        # the input takes the observed focal populations, the target takes the ones being predicteds
        # stacking on axis 1 keeps rows simulation major, so every row traces back to one simulation
        X = np.stack(
            [data[:, self.input_indices, start : start + w] for start in starts],
            axis=1,
        ).reshape(-1, self.n_input_focal, w)

        y = np.stack(
            [data[:, self.target_indices, start + w : start + (2 * w)] for start in starts],
            axis=1,
        ).reshape(-1, self.n_target_focal, w)

        assert X.shape == (data.shape[0] * len(starts), self.n_input_focal, w)
        assert y.shape == (data.shape[0] * len(starts), self.n_target_focal, w)

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        )


    def _normalize_curves(self, curves):
        '''divide each curve by its own maximum, returning the normalised curves and the maxima'''
        # curves has shape (n_curves, window_size)
        maxima = curves.max(axis=1, keepdims=True)

        # extinct focal populations are all zero, map them to ones exactly like get_latents does
        normalized = np.ones_like(curves)
        np.divide(
            curves,
            maxima,
            out=normalized,
            where=maxima > 0,
        )

        return normalized, maxima


    def _pca_inputs(self, curves):
        '''return the curves in the space the PCA is both fit and applied in'''
        # PCA is always fit and applied on max normalised curves, mirroring how the VAE encoder
        # only ever sees normalised curves, so append_max only ever controls whether the maximum
        # is added as an extra feature and never the PCA space itself
        return self._normalize_curves(curves)


    def _fit_pca(self, X_train_raw, y_train_raw,):
        '''fit PCA functions with training data, one focal curve is one PCA sample'''

        needs_input_pca = (self.input_type == 'pca')
        needs_target_pca = (self.target_type == 'pca')

        if not (needs_input_pca or needs_target_pca):
            return

        # PCA is fit per focal curve so it consumes curves the same way the latent encoder does
        X_curves, _ = self._pca_inputs(
            X_train_raw.reshape(-1, self.window_size)
        )

        y_curves, _ = self._pca_inputs(
            y_train_raw.reshape(-1, self.window_size)
        )

        # shared PCA space
        if (self.input_type == 'pca' and self.target_type == 'pca'):
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
        '''returns windows in specified formats, with shape (n_windows, n_focal * n_features)'''
        # windows has shape (n_windows, n_focal, window_size)
        n_windows = windows.shape[0]

        if representation == 'raw':
            return windows.reshape(n_windows, -1).astype(np.float32)

        if representation == 'latent':
            features = self._encode_latent(
                windows.reshape(-1, self.window_size)
            )

        elif representation == 'pca':
            features = self._encode_pca(
                windows.reshape(-1, self.window_size),
                side,
            )

        else:
            raise ValueError(
                f'Unknown representation type: {representation}'
            )

        # regroup the per focal curve features back into one row per window
        features = features.reshape(n_windows, -1).astype(np.float32)
        assert features.shape[0] == n_windows

        return features


    def _encode_latent(self, curves):
        '''latent encoder wrapper, get_latents normalises each curve and returns its max last'''
        # get_latents returns (n_curves, z_dim + 1) where the final column is the curve maximum
        features = get_latents(
            z=curves,
            model=self.encoder,
            z_dim=self.z_dim,
            batch_size=4096,
        )

        if not self.append_max:
            # the encoder only ever sees max normalised curves, so dropping the maximum
            # discards every trace of absolute abundance
            features = features[:, :-1]

        return features


    def _encode_pca(self, curves, side):
        '''PCA encoder wrapper'''

        pca_input, maxima = self._pca_inputs(curves)

        if self.shared_pca is not None:
            components = self.shared_pca.transform(pca_input)

        elif side == 'input':
            components = self.input_pca.transform(pca_input)

        elif side == 'target':
            components = self.target_pca.transform(pca_input)

        else:
            raise ValueError(f'Unknown side, not input or target: {side}')

        if not self.append_max:
            return components

        return np.concatenate(
            [components, maxima],
            axis=1,
        )
