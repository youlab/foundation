from torch import nn


class MLP(nn.Module):
    name = "mlp"

    def __init__(
        self,
        in_features,
        hidden_dim,
        out_features,
        num_layers,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        assert num_layers >= 3, ValueError(
            f"num_layers must be >= 3, but is {num_layers}"
        )
        modules = [
            nn.Linear(
                in_features=in_features,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
        ]
        for i in range(num_layers - 2):
            modules.append(
                nn.Linear(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                )
            )
            modules.append(nn.ReLU())
        modules.append(
            nn.Linear(
                in_features=hidden_dim,
                out_features=out_features,
            )
        )
        self.net = nn.Sequential(*modules)

    def forward(
        self,
        x,
    ):
        return self.net(x)
