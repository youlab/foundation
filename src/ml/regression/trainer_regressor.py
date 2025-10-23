import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def warmup_schedule(
    epoch,
):
    warmup_epochs = 10
    return (epoch / warmup_epochs) ** 2


class TrainerRegressor:
    def __init__(
        self,
        regressor,
        dataset_train,
        dataset_test,
        epochs=20000,
        patience=50,
        warmup_epochs=20,
        weight_decay=1e-5,
        lr=2e-4,
        min_lr_mult=1e-5,
        gamma=0.996,
        test_frequency=10,
        batch_size=4096,
    ):
        self.regressor = regressor
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.test_frequency = test_frequency
        self.patience = patience

        self.x_train = torch.tensor(dataset_train.x[:, :, :-1]).to(device).float()
        self.x_test = torch.tensor(dataset_test.x[:, :, :-1]).to(device).float()

        self.y_train = dataset_train.x[:, :, -1]
        self.y_test = dataset_test.x[:, :, -1]

        self.loader_train = DataLoader(
            dataset_train,
            batch_size,
        )
        self.loader_test = DataLoader(
            dataset_test,
            batch_size,
        )

        self.train_loss = []
        self.test_loss = []

        self.min_lr = lr * min_lr_mult

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.regressor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler_warmup = LambdaLR(
            self.optimizer,
            lr_lambda=warmup_schedule,
        )

        self.scheduler_exponential = ExponentialLR(
            self.optimizer,
            gamma=gamma,
        )

    def train(
        self,
    ):
        epochs_no_improve = 0
        best_test_loss = np.inf
        with logging_redirect_tqdm():
            for epoch in trange(self.epochs):
                self.regressor.train()
                running_loss = []

                self.optimizer.zero_grad()
                for x, tgt in self.loader_train:
                    out = self.regressor(x=x.to(device).float())
                    loss = self.criterion(
                        out,
                        tgt.to(device).float(),
                    )
                    loss.backward()
                    self.optimizer.step()
                    running_loss.append(loss.item() / x.size(0))

                if epoch > self.warmup_epochs:
                    self.scheduler_exponential.step()
                    self.optimizer.param_groups[0]["lr"] = max(
                        self.optimizer.param_groups[0]["lr"],
                        self.min_lr,
                    )
                else:
                    self.scheduler_warmup.step()

                self.train_loss.append(np.mean(running_loss))

                if epoch % self.test_frequency == 0:
                    self.regressor.eval()
                    running_loss = []
                    with torch.no_grad():
                        for x, tgt in self.loader_test:
                            out = self.regressor(x=x.to(device).float())
                            loss = self.criterion(
                                out,
                                tgt.to(device).float(),
                            )
                            running_loss.append(loss.item() / x.size(0))
                    self.test_loss.append(np.mean(running_loss))

                    if self.test_loss[-1] < best_test_loss:
                        best_test_loss = self.test_loss[-1]
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve == self.patience:
                        break

                else:
                    self.test_loss.append(self.test_loss[-1])
                    epochs_no_improve += 1

        return self.test_loss[-1]

    def predict_train(self):
        self.regressor.eval()
        with torch.no_grad():
            return self.regressor(self.x_train)

    def predict_test(self):
        self.regressor.eval()
        with torch.no_grad():
            return self.regressor(self.x_test)
