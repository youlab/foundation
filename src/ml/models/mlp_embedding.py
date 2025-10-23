import torch
from torch import nn


class MLPEmbedding(nn.Module):
    name = "mlp_embedding"

    def __init__(
        self,
        in_features,
        hidden_dim,
        out_features,
        num_layers,
        idx_embeddings,
        num_embeddings,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.idx_embeddings = idx_embeddings
        assert num_layers >= 3, ValueError(
            f"num_layers must be >= 3, but is {num_layers}"
        )

        self.embedding = nn.Embedding(
            num_embeddings=int(num_embeddings) + 1,
            embedding_dim=1,
            max_norm=True,
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
        x_embed_in = x[:, self.idx_embeddings].detach().long().reshape(-1, 1)
        x_embed_out = self.embedding(x_embed_in)
        return self.net(
            torch.concatenate(
                (
                    x[:, : self.idx_embeddings],
                    x_embed_out[:, :, 0],
                    x[:, self.idx_embeddings + 1 :],
                ),
                axis=1,
            )
        )
