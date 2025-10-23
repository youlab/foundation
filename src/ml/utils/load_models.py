import json
from joblib import load

import torch
from huggingface_hub import hf_hub_download

from config import (
    MODEL_TYPE,
    Z_DIM,
)
from ml.models.autoencoder7x import Autoencoder7XModel
from ml.models.mlp_network_model import MLPNetworkModel
from ml.models.microbert_curve_reducer import MicroBERTCurveReducerModel
from ml.models.vae_bottleneck import VAEBottleneckModel
from ml.utils.formatting import num_to_str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_best_models():
    return [
        ("A7X", 2,),
        ("A7X", 4,),
        ("A7X", 6,),
        ("A7X", 8,),
        ("A7X", 12,),
        ("A7X", 16,),
        ("A7X", 20,),
        ("A7X", 24,),
        ("A7X", 32,),
        ("MCR", 2,),
        ("MCR", 4,),
        ("MCR", 6,),
        ("MCR", 8,),
        ("MCR", 12,),
        ("MCR", 16,),
        ("MCR", 20,),
        ("MCR", 24,),
        ("MCR", 32,),
        ("MNM", 2,),
        ("MNM", 4,),
        ("MNM", 6,),
        ("MNM", 8,),
        ("MNM", 12,),
        ("MNM", 16,),
        ("MNM", 20,),
        ("MNM", 24,),
        ("MNM", 32,),
        ("PR", 2,),
        ("PR", 4,),
        ("PR", 6,),
        ("PR", 8,),
        ("PR", 12,),
        ("PR", 16,),
        ("PR", 20,),
        ("PR", 24,),
        ("PR", 32,),
        ("VB", 2,),
        ("VB", 4,),
        ("VB", 8,),
        ("VB", 16,),
        ("VB", 32,),
    ]


def load_default_model():
    return load_model(
        model_type=MODEL_TYPE,
        z_dim=Z_DIM,
    )


def load_pr(
    z_dim,
):
    model_path = hf_hub_download(
        repo_id=f"you-lab/pca-reducer-{num_to_str(n=z_dim)}",
        filename=f"pca_{z_dim}.joblib",
        repo_type="model"
    )
    return load(model_path)


def load_model(
    model_type,
    z_dim,
):
    if model_type == "A7X":
        model = Autoencoder7XModel.from_pretrained(f"you-lab/autoencoder7x-{num_to_str(n=z_dim)}")
    elif model_type == "MNM":
        model = MLPNetworkModel.from_pretrained(f"you-lab/mlp-network-model-{num_to_str(n=z_dim)}")
    elif model_type == "MCR":
        model = MicroBERTCurveReducerModel.from_pretrained(f"you-lab/microbert-curve-reducer-{num_to_str(n=z_dim)}")
    elif model_type == "VB":
        model = VAEBottleneckModel.from_pretrained(f"you-lab/vae-bottleneck-{num_to_str(n=z_dim)}")
    elif model_type == "PR":
        return load_pr(z_dim=z_dim)
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")
    model.to(device)
    model.eval()
    return model
