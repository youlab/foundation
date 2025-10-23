import torch
from torch import nn


class NeuralODE(nn.Module):
    def __init__(
        self,
        hidden_features=128,
    ):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(2, hidden_features),
            nn.Tanh(),
            nn.Linear(
                hidden_features,
                hidden_features,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_features,
                hidden_features,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_features,
                hidden_features,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_features,
                hidden_features,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_features,
                hidden_features,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_features,
                1,
            ),
        )

        for m in self.linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        return self.linear(z)


def ode_system(t, y, model, mu):
    return model(
        t,
        torch.cat((y.view(-1, 1), mu.view(-1, 1)), dim=1),
    ).flatten()
