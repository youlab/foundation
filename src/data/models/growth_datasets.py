"""
This file contains the code to create the Train and Test Datasets.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DIR_DATA
from data.normalization_functions.utils import robust_scaling

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
            self.mu[idx],
            self.t_lag[idx],
            self.y0[idx],
        )


class GrowthTrainDataset(Dataset):
    """
    This class is the test data for training a machine learning more on growth curves.
    """
    def __init__(
        self,
        file_prefix,
        train_ratio=0.8,
    ):
        """
        This method initializes the class with the growth curve data.
        ----------
        PARAMETERS
        file_prefix -> str : file prefix for the file to load with the data.
        train_ratio -> float : value from 0 to 1 to indicate how much of the data should be used for training.
        """
        self.dir = DIR_DATA / "simulation" / "lingchong_eqn"

        data = np.load(self.dir / f"{file_prefix}.npz")
        self.y = data["y"]
        self.y_max = data["y_max"]
        self.y0 = data["y0"]

        self.mu = np.load(
            self.dir / f"{file_prefix}_alpha.npy",
        )

        self.t_lag = np.load(
            self.dir / f"{file_prefix}_t_lag.npy",
        )

        self.y_train = self.y[: int(self.y.shape[0] * train_ratio), :]
        self.y_test = self.y[int(self.y.shape[0] * train_ratio) :, :]

        self.y_max_train = self.y_max[: int(self.y.shape[0] * train_ratio), :]
        self.y_max_test = self.y_max[int(self.y.shape[0] * train_ratio) :, :]

        self.y0_train = self.y0[:int(self.y.shape[0] * train_ratio)]
        self.y0_test = self.y0[int(self.y.shape[0] * train_ratio):]

        self.mu_train = self.mu[: int(self.y.shape[0] * train_ratio)]
        self.mu_test = self.mu[int(self.y.shape[0] * train_ratio):]

        self.mu_train, self.mu_test = robust_scaling(
            x_train=self.mu_train,
            x_test=self.mu_test,
        )

        self.t_lag_train = self.t_lag[: int(self.y.shape[0] * train_ratio)]
        self.t_lag_test = self.t_lag[int(self.y.shape[0] * train_ratio):]

        self.t_lag_train, self.t_lag_test = robust_scaling(
            x_train=self.t_lag_train,
            x_test=self.t_lag_test,
        )

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
            self.mu_train[idx],
            self.t_lag_train[idx],
            self.y0_train[idx],
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
        )


def get_data_loaders(
    file_prefix,
    batch_size=512,
    train_ratio=0.8,
):
    """
    This function initializes a dataset and returns the dataloaders for the training.
    ----------
    PARAMETERS
    file_prefix -> str : file prefix for the file to load with the data.
    batch_size -> int : the number of samples to return in each call of the dataloader.
    """
    dataset = GrowthTrainDataset(
        file_prefix=file_prefix,
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
