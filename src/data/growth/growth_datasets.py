"""
This file contains the code to create the Train and Test Datasets.

These data files from the mu_model directory in the growth_parameters repository.
They were created using generate_mu_model_data.py.
"""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.normalization_functions.utils import (
    interpolate_y,
    generate_train_test_idx,
    robust_scaling,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrowthTestDataset(Dataset):
    """
    This class is the test data for training a machine learning more on growth curves.
    """

    def __init__(
        self,
        y,
        y_max,
        mu,
        t_lag,
        y0,
        media,
        strain,
        phase,
    ):
        """
        This method initializes the class with the growth curve data.
        ----------
        PARAMETERS
        y -> torch.tensor : growth curves arranged in shape n_samples x n_timepoints
        """
        self.y = y
        self.y_max = y_max
        self.mu = mu
        self.t_lag = t_lag
        self.y0 = y0
        self.media = media
        self.strain = strain
        self.phase = phase

    def __len__(self):
        """
        This method returns the number of samples in the test datset.
        """
        return self.y.shape[0]

    def __getitem__(
        self,
        idx,
    ):
        """
        This method returns the item from the test dataset.
        ----------
        PARAMETERS
        idx -> int : index to return
        """
        return (
            self.y[idx, :],
            self.y_max[idx, :],
            self.mu[idx, :],
            self.t_lag[idx, :],
            self.y0[idx, :],
            self.media[idx, :],
            self.strain[idx, :],
            self.phase[idx, :],
        )


class GrowthTrainDataset(Dataset):
    """
    This class is the test data for training a machine learning more on growth curves.
    """
    def __init__(
        self,
        seq_len=128,
        train_ratio=0.8,
    ):
        """
        This method initializes the class with the growth curve data.
        ----------
        PARAMETERS
        train_ratio -> float : value from 0 to 1 to indicate how much of the data should be used for training.
        """
        print("train_ratio in dataset", train_ratio)
        self.dir = (
            Path.home() / "foundations" / "src" / "data" / "growth"
        )

        for _, _, files in os.walk(self.dir):
            break

        self.y = None
        self.mu = None
        self.t_lag = None
        self.y0 = None
        self.media = None
        self.strain = None
        self.phase = None
        for fn in files:
            if fn.find("_t_lag.npz") != -1:
                if self.t_lag is None:
                    self.t_lag = np.load(self.dir / fn)["t_lag"]
                else:
                    self.t_lag = np.concatenate((
                        self.t_lag,
                        np.load(self.dir / fn)["t_lag"],
                    ))
            elif fn.find(".npz") != -1:
                data = np.load(self.dir / fn, allow_pickle=True,)
                y_temp = data["y_raw"]
                if y_temp.shape[1] == seq_len:
                    pass
                elif y_temp.shape[1] >= seq_len:
                    y_temp = y_temp[:, :seq_len]
                else:
                    y_temp = interpolate_y(y=y_temp, interp_len=seq_len,)
                if self.y is None:
                    self.y = y_temp
                    self.mu = data["mu"]
                    self.y0 = data["y0"]
                    self.media = data["media"]
                    self.strain = data["strain"]
                    self.phase = data["phase"]
                else:
                    self.y = np.concatenate((
                        self.y,
                        y_temp,
                    ))
                    self.mu = np.concatenate((
                        self.mu,
                        data["mu"],
                    ))
                    self.y0 = np.concatenate((
                        self.y0,
                        data["y0"],
                    ))
                    self.media = np.concatenate((
                        self.media,
                        data["media"],
                    ))
                    self.strain = np.concatenate((
                        self.strain,
                        data["strain"],
                    ))
                    self.phase = np.concatenate((
                        self.phase,
                        data["phase"],
                    ))

        train_idx, test_idx = generate_train_test_idx(
            n=self.y.shape[0],
            train_ratio=train_ratio,
        )

        self.mu = self.mu.reshape(-1, 1)
        self.t_lag = self.t_lag.reshape(-1, 1)
        self.y0 = self.y0.reshape(-1, 1)
        
        self.y_max = self.y.max(axis=1).reshape(-1, 1)
        self.y = self.y / self.y_max
        
        self.y_train = self.y[train_idx, :]
        self.y_test = self.y[test_idx, :]

        self.y_max_train = self.y_max[train_idx, :]
        self.y_max_test = self.y_max[test_idx, :]

        self.mu_train, self.mu_test = robust_scaling(
            x_train=self.mu[train_idx, :],
            x_test=self.mu[test_idx, :],
        )

        self.t_lag[self.t_lag < 0] = 0

        self.t_lag_train, self.t_lag_test = robust_scaling(
            x_train=self.t_lag[train_idx, :],
            x_test=self.t_lag[test_idx, :],
        )

        self.y0_train = self.y0[train_idx, :] / self.y_max[train_idx, :]
        self.y0_test = self.y0[test_idx, :] / self.y_max[test_idx, :]

        self.media_unique = np.unique(self.media)
        self.media_train = np.zeros((train_idx.shape[0], self.media_unique.shape[0]))
        self.media_test = np.zeros((test_idx.shape[0], self.media_unique.shape[0]))
        for i, media in enumerate(self.media_unique):
            self.media_train[self.media[train_idx] == media, i] = 1
            self.media_test[self.media[test_idx] == media, i] = 1
        
        self.strain_unique = np.unique(self.strain)
        self.strain_train = np.zeros((train_idx.shape[0], self.strain_unique.shape[0]))
        self.strain_test = np.zeros((test_idx.shape[0], self.strain_unique.shape[0]))
        for i, strain in enumerate(self.strain_unique):
            self.strain_train[self.strain[train_idx] == strain, i] = 1
            self.strain_test[self.strain[test_idx] == strain, i] = 1

        self.phase_unique = np.unique(self.phase)
        self.phase_train = np.zeros((train_idx.shape[0], self.phase_unique.shape[0]))
        self.phase_test = np.zeros((test_idx.shape[0], self.phase_unique.shape[0]))
        for i, phase in enumerate(self.phase_unique):
            self.phase_train[self.phase[train_idx] == phase, i] = 1
            self.phase_test[self.phase[test_idx] == phase, i] = 1

        self.y_train = (
            torch.tensor(
                self.y_train,
            )
            .view(
                self.y_train.shape[0],
                1,
                self.y_train.shape[1],
            )
            .float()
            .to(device)
        )

        self.y_test = (
            torch.tensor(self.y_test)
            .view(
                self.y_test.shape[0],
                1,
                self.y_test.shape[1],
            )
            .float()
            .to(device)
        )

    def __len__(self):
        """
        This method returns the number of samples in the train datset.
        """
        return self.y_train.shape[0]

    def __getitem__(
        self,
        idx,
    ):
        """
        This method returns the item from the train dataset.
        ----------
        PARAMETERS
        idx -> int : index to return
        """
        return (
            self.y_train[idx, :],
            self.y_max_train[idx, :],
            self.mu_train[idx, :],
            self.t_lag_train[idx, :],
            self.y0_train[idx, :],
            self.media_train[idx, :],
            self.strain_train[idx, :],
            self.phase_train[idx, :],
        )

    def get_test_dataset(self):
        """
        This method returns the test dataset associated with this train dataset.
        """
        return GrowthTestDataset(
            y=self.y_test,
            y_max=self.y_max_test,
            mu=self.mu_test,
            t_lag=self.t_lag_test,
            y0=self.y0_test,
            media=self.media_test,
            strain=self.strain_test,
            phase=self.phase_test,
        )


def get_experimental_data_loaders(
    seq_len=128,
    batch_size=512,
    train_ratio=0.8,
):
    """
    This function initializes a dataset and returns the dataloaders for the training.
    ----------
    PARAMETERS
    batch_size -> int : the number of samples to return in each call of the dataloader.
    """
    print("train_ratio", train_ratio)
    dataset = GrowthTrainDataset(
        seq_len=seq_len,
        train_ratio=train_ratio,
    )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )

    test_loader = DataLoader(
        dataset=dataset.get_test_dataset(),
        batch_size=batch_size,
    )

    return dataset, train_loader, test_loader
