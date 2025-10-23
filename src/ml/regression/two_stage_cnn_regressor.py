import numpy as np
import torch

from ml.models.custom_regressor import (
    CNNRegressor,
    TrainerRegressor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoStageCnnRegressor:
    def __init__(
        self,
        in_channels,
        hidden_dim,
        z_dims,
        extra_features,
        num_layers,
        out_features,
        x_train,
        x_test,
        y_train,
        y_test,
    ):
        self.regressor1 = CNNRegressor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            z_dims=z_dims,
            extra_features=extra_features,
            num_layers=num_layers,
            out_features=1,
        ).to(device)

        self.regressor2 = CNNRegressor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            z_dims=z_dims,
            extra_features=extra_features + 1,
            num_layers=num_layers,
            out_features=1,
        ).to(device)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, x, tgt,):
        assert tgt.shape[1] == 2, ValueError("Target should have 2 variables.")

        self.trainer1 = TrainerRegressor(
            regressor=self.regressor1,
            x_train=self.x_train,
            x_test=self.x_test,
            y_train=self.y_train[:, 0].reshape(-1, 1),
            y_test=self.y_test[:, 0].reshape(-1, 1),
            epochs=2000,
            patience=50,
            warmup_epochs=20,
            weight_decay=1e-5,
            lr=2e-5,
            min_lr_mult=1e-5,
            gamma=0.996,
            test_frequency=10,
        )

        self.trainer1.train()

        x2_train = np.concatenate(
            (
                self.x_train,
                self.trainer1.predict_train().cpu().detach().numpy(),
            ),
            axis=1,
        )

        x2_test = np.concatenate(
            (
                self.x_test,
                self.trainer1.predict_test().cpu().detach().numpy(),
            ),
            axis=1,
        )

        self.trainer2 = TrainerRegressor(
            regressor=self.regressor2,
            x_train=x2_train,
            x_test=x2_test,
            y_train=self.y_train[:, 1].reshape(-1, 1),
            y_test=self.y_test[:, 1].reshape(-1, 1),
            epochs=2000,
            patience=50,
            warmup_epochs=20,
            weight_decay=1e-5,
            lr=2e-5,
            min_lr_mult=1e-5,
            gamma=0.996,
            test_frequency=10,
        )

        self.trainer2.train()

    def predict(self, x,):
        if x.shape[0] == self.x_train.shape[0]:
            return np.concatenate(
                (
                    self.trainer1.predict_train().cpu().detach().numpy(),
                    self.trainer2.predict_train().cpu().detach().numpy(),
                ),
                axis=1,
            )
        elif x.shape[0] == self.x_test.shape[0]:
            return np.concatenate(
                (
                    self.trainer1.predict_test().cpu().detach().numpy(),
                    self.trainer2.predict_test().cpu().detach().numpy(),
                ),
                axis=1,
            )
