"""
This file contains the code to create the Train and Test Datasets.
"""
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from data.utils import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    """
    This class is the test data for training a machine learning more on growth curves.
    """

    def __init__(
        self,
        y,
    ):
        """
        This method initializes the class with the growth curve data.
        ----------
        PARAMETERS
        y -> torch.tensor : growth curves arranged in shape n_samples x n_timepoints
        """
        self.y = y

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
        return self.y[idx, :]


class TrainDataset(Dataset):
    """
    This class is the test data for training a machine learning more on growth curves.
    """

    def __init__(
        self,
        category,
    ):
        """
        This method initializes the class with the growth curve data.
        ----------
        PARAMETERS
        input_length -> str : file prefix for the file to load with the data.
        """
        (
            self.y_train,
            self.y_test,
        ) = get_data(
            category=category,
            return_split=True,
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
        return self.y_train[idx, :]

    def get_test_dataset(self):
        """
        This method returns the test dataset associated with this train dataset.
        """
        return TestDataset(y=self.y_test)


def get_data_loaders(
    category,
    batch_size=256,
):
    """
    This function initializes a dataset and returns the dataloaders for the training.
    ----------
    PARAMETERS
    file_prefix -> str : file prefix for the file to load with the data.
    batch_size -> int : the number of samples to return in each call of the dataloader.
    """
    dataset = TrainDataset(category=category)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )

    test_loader = DataLoader(
        dataset=dataset.get_test_dataset(),
        batch_size=batch_size,
    )

    return train_loader, test_loader
