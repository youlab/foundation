import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin

from ml.models.mlp import MLP


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


class MLPNetworkModel(
    nn.Module,
    PyTorchModelHubMixin,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        config,
    ):
        super().__init__()
        input_dim = config.get("input_dim")
        hidden_dim = config.get("hidden_dim")
        num_layers = config.get("num_layers")
        z_dim = config.get("z_dim")

        self.encoder = nn.Sequential(
            MLP(
                in_features=input_dim,
                hidden_dim=hidden_dim,
                out_features=z_dim,
                num_layers=num_layers,
            ),
            nn.ReLU(),
        )

        self.decoder = MLP(
            in_features=z_dim,
            hidden_dim=hidden_dim,
            out_features=input_dim,
            num_layers=num_layers,
        )

        self.mu = nn.Linear(
            in_features=z_dim,
            out_features=z_dim,
        )

        self.logvar = nn.Linear(
            in_features=z_dim,
            out_features=z_dim,
        )

        self.apply(init_weights)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x,
    ):
        encoded = self.encoder(x)
        mean = self.mu(encoded)
        logvar = self.logvar(encoded)
        if self.training:
            z = self.reparameterize(mean, logvar)
            reconstruction = self.decoder(z)
        else:
            reconstruction = self.decoder(mean)
        return reconstruction, mean, logvar
