import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


class Autoencoder7XModel(
    nn.Module,
    PyTorchModelHubMixin,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        config,
    ):
        super().__init__()
        input_length = config.get("input_length")
        channel2 = config.get("channel2")
        channel3 = config.get("channel3")
        channel4 = config.get("channel4")
        reduction = config.get("reduction")
        z_dim = config.get("z_dim")

        if reduction not in {
            2,
            4,
            8,
            16,
            32,
            64,
        }:
            raise ValueError(
                f"Parameter 'reduction' should be 2, 4, 8, 16, 32, or 64, but it is {reduction}"
            )

        self.reduction = reduction
        self.z_dim = z_dim
        self.input_length = input_length
        self.hidden_dim = 64

        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.dilation = 1

        self.k_flat = 3
        self.s_flat = 1

        self.channel1 = 1
        self.channel2 = channel2
        self.channel3 = channel3
        self.channel4 = channel4

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.channel1,
                out_channels=self.channel2,
                kernel_size=self.k_flat if reduction <= 32 else self.kernel_size,
                stride=self.s_flat if reduction <= 32 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.channel2,
                out_channels=self.channel3,
                kernel_size=self.k_flat if reduction <= 8 else self.kernel_size,
                stride=self.s_flat if reduction <= 8 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.channel3,
                out_channels=self.channel4,
                kernel_size=self.k_flat if reduction <= 4 else self.kernel_size,
                stride=self.s_flat if reduction <= 4 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.channel4,
                out_channels=self.channel4,
                kernel_size=self.k_flat if reduction <= 2 else self.kernel_size,
                stride=self.s_flat if reduction <= 2 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.channel4,
                out_channels=self.channel3,
                kernel_size=self.k_flat if reduction <= 16 else self.kernel_size,
                stride=self.s_flat if reduction <= 16 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.channel3,
                out_channels=self.channel2,
                kernel_size=self.k_flat,
                stride=self.s_flat,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(
                in_features=self.channel2 * int(self.input_length / self.reduction) * 2,
                out_features=self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.z_dim,
            ),
        )

        self.logvar = nn.Sequential(
            nn.Linear(
                in_features=self.channel2 * int(self.input_length / self.reduction) * 2,
                out_features=self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.z_dim,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=self.z_dim,
                out_features=int(self.input_length / self.reduction),
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=self.channel1,
                out_channels=self.channel2,
                kernel_size=self.k_flat,
                stride=self.s_flat,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=self.channel2,
                out_channels=self.channel3,
                kernel_size=self.k_flat if reduction <= 16 else self.kernel_size,
                stride=self.s_flat if reduction <= 16 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=self.channel3,
                out_channels=self.channel4,
                kernel_size=self.k_flat if reduction <= 2 else self.kernel_size,
                stride=self.s_flat if reduction <= 2 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=self.channel4,
                out_channels=self.channel4,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=self.channel4,
                out_channels=self.channel3,
                kernel_size=self.k_flat if reduction <= 4 else self.kernel_size,
                stride=self.s_flat if reduction <= 4 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=self.channel3,
                out_channels=self.channel2,
                kernel_size=self.k_flat if reduction <= 8 else self.kernel_size,
                stride=self.s_flat if reduction <= 8 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=self.channel2,
                out_channels=self.channel1,
                kernel_size=self.k_flat if reduction <= 32 else self.kernel_size,
                stride=self.s_flat if reduction <= 32 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
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
        encoded = self.encoder(x).reshape(x.shape[0], -1)
        mean = self.mu(encoded)
        logvar = self.logvar(encoded)
        if self.training:
            z = self.reparameterize(mean, logvar)
            reconstruction = self.decoder(z.reshape(z.shape[0], 1, z.shape[1]))
        else:
            reconstruction = self.decoder(mean.reshape(mean.shape[0], 1, mean.shape[1]))
        return reconstruction, mean, logvar

    def preprocess_data(self, x):
        return x.float().to(self.device)

    def calculate_loss(
        self,
        x,
        recon,
        criterion,
        alpha=None,
        mu=None,
        logvar=None,
    ):
        recon_loss = criterion(recon, x)
        if not self.training:
            return recon_loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + alpha * kl_loss
        return loss
