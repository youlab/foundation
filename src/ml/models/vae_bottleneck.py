import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin


class VAEBottleneckModel(
    nn.Module,
    PyTorchModelHubMixin,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        config,
    ):
        super().__init__()
        channel2 = config.get("channel2")
        channel3 = config.get("channel3")
        channel4 = config.get("channel4")
        reduction = config.get("reduction")

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
                kernel_size=self.k_flat if reduction <= 2 else self.kernel_size,
                stride=self.s_flat if reduction <= 2 else self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.channel4,
                out_channels=self.channel4,
                kernel_size=self.k_flat if reduction <= 4 else self.kernel_size,
                stride=self.s_flat if reduction <= 4 else self.stride,
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
            nn.Conv1d(
                in_channels=self.channel2,
                out_channels=self.channel1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
        )

        self.logvar = nn.Sequential(
            nn.Conv1d(
                in_channels=self.channel2,
                out_channels=self.channel1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ),
        )

        self.decoder = nn.Sequential(
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
