import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from config import SEQ_LEN


class MicroBERTCurveReducerModel(
    nn.Module,
    PyTorchModelHubMixin,
):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        dimension = config.get("dimension")
        n_heads = config.get("n_heads")
        n_layers = config.get("n_layers")
        self.embedding = nn.Linear(
            SEQ_LEN,
            dimension,
        ) 
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=n_heads,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.mlm_head = nn.Linear(
            dimension,
            SEQ_LEN,
        )
        
    def forward(self, x):
        x = self.embedding(x)
        encoder_output = self.transformer_encoder(x)
        mlm_output = self.mlm_head(encoder_output)
        return encoder_output, mlm_output
   
    def get_embedding(self, x):
        with torch.no_grad():
            x = self.embedding(x)
            encoder_output = self.transformer_encoder(x)
            return encoder_output[:, -1, :]

    def encode(
        self,
        x,
    ):
        with torch.no_grad():
            return self.embedding(x)

    def mu(
        self,
        x,
    ):
        with torch.no_grad():
            encoder_output = self.transformer_encoder(x)
            return encoder_output[:, -1, :]
