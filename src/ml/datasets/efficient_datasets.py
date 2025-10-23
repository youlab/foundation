import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_efficient_data_loaders(
    x_train,
    x_test,
    batch_size=2_048,
):
    """
    This function initializes a dataset and returns the dataloaders for the training.
    ----------
    PARAMETERS
    x_train -> np.ndarray : numpy array of shape [n_samples, SEQ_LEN]
    x_test -> np.ndarray : numpy array of shape [n_samples, SEQ_LEN]
    batch_size -> int : the number of samples to return in each call of the dataloader.
    """
    x_train = (
        torch.tensor(
            x_train,
        )
        .view(
            x_train.shape[0],
            1,
            x_train.shape[1],
        )
        .float()
        .to(device)
    )
    x_test = (
        torch.tensor(
            x_test,
        )
        .view(
            x_test.shape[0],
            1,
            x_test.shape[1],
        )
        .float()
        .to(device)
    )
    
    train_loader = DataLoader(
        x_train,
        batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        x_test,
        batch_size,
    )

    return train_loader, test_loader
