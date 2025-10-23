import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from ml.models.growth_model import GrowthModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_train_test_idx(n):
    if not isinstance(n, int):
        return TypeError(f"n should be int but is {type(n)}")
    
    N_SET = 1000
    TRAIN_SIZE = 0.8
    n_loops = n // N_SET
    idx = np.arange(n)
    n_train = int(n * TRAIN_SIZE)
    n_test = n - n_train
    n_train_loop = int(N_SET * TRAIN_SIZE)
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
                train_size=TRAIN_SIZE,
            )
        i_add = n_loops * N_SET
    else:
        random_state = -1
        i_add = 0

    if n > (n_loops * N_SET):
        print(idx_train.shape, idx_test.shape)
        print(i_add, TRAIN_SIZE, idx_train[int(i_add * TRAIN_SIZE):].shape, idx_test[int(i_add * (1-TRAIN_SIZE)):].shape, int(i_add * TRAIN_SIZE), int(i_add * (1-TRAIN_SIZE)))
        n_train_loop = int((n % N_SET) * TRAIN_SIZE)
        n_test_loop = (n % N_SET) - n_train_loop
    
        idx_train[-n_train_loop:], idx_test[-n_test_loop:] = train_test_split(
            idx[i_add:],
            random_state=random_state + 1,
            train_size=TRAIN_SIZE,
        )
    
    return idx_train, idx_test


def get_true(
    n,
    growth_model,
):
    model_true = GrowthModel(n=n)
    sol = model_true.solve_ode()

    z0 = torch.tensor(model_true.y0).float().to(device)
    mu = torch.tensor(growth_model.mu).float().to(device)

    z_true = torch.tensor(sol.y.T).float().to(device)
    t = torch.tensor(sol.t).float().to(device)

    return t, z0, mu, z_true


def GrowthDataset(Dataset):
    def __init__(
        self,
        n,
    ):
        super().__init__()

        self.t, z0, mu, z = get_true(n=n)

        train_idx, test_idx = generate_train_test_idx(n=n)

        self.z0_train = z0[train_idx]
        self.z_train = z[train_idx, :]
        self.mu_train = mu[train_idx]

        self.z0_test = z0[test_idx]
        self.z_test = z[test_idx, :]
        self.mu_test = mu[test_idx]

    def __len__(self):
        return self.z0_train.shape[0]
    
    def __getitem__(
        self,
        idx,
    ):
        return self.z0_train[idx], self.mu_train[idx], self.z_train[idx], self.t

    def get_test_dataset(self):
        return GrowthDatasetTest(
            z0=self.z0_test,
            mu=self.mu_test,
            z=self.z_test,
            t=self.t,
        )


def GrowthDatasetTest(Dataset):
    def __init__(
        self,
        z0,
        mu,
        z,
        t,
    ):
        super().__init__()

        self.z0 = z0
        self.mu = mu
        self.z = z
        self.t = t

    def __len__(self):
        return self.z0.shape[0]
    
    def __getitem__(
        self,
        idx,
    ):
        return self.z0[idx], self.mu[idx], self.z[idx], self.t


def get_data_loaders(
    n=10_000,
    batch_size=256,
):
    train_dataset = GrowthDataset(n=n)
    test_dataset = train_dataset.get_test_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    return train_loader, test_loader
