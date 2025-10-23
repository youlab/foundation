import torch
from torch import nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


class CNNRegressor(nn.Module):
    def __init__(
        self,
        in_channels,
        n_samples,
        hidden_dim,
        z_dims,
        num_layers=3,
    ):
        super().__init__()
        self.n_samples = n_samples

        kernel_size = 3
        stride = 1
        dilation = 1
        padding = 1

        modules = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            ),
            nn.ReLU(),
        ]
        for i in range(num_layers - 2):
            modules.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                )
            )
            modules.append(nn.ReLU())

        kernel_size = 3
        stride = 1
        dilation = 1
        padding = 1

        modules.append(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
        )
        modules.append(nn.ReLU())

        self.net = nn.Sequential(*modules)

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim * z_dims * n_samples,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=n_samples,
            ),
        )

        self.apply(init_weights)

    def forward(
        self,
        x,
    ):
        encoded = self.net(x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])).reshape(x.shape[0], -1)
        return self.fc(encoded)
