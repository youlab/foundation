import torch
from sklearn.decomposition import IncrementalPCA

from ml.config import (
    CONFIG_A7X,
    CONFIG_MCR,
    CONFIG_MNM,
    CONFIG_VB,
)
from ml.models.autoencoder7x import Autoencoder7XModel
from ml.models.mlp_network_model import MLPNetworkModel
from ml.models.microbert_curve_reducer import MicroBERTCurveReducerModel
from ml.models.vae_bottleneck import VAEBottleneckModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_untrained_model(
    model_type,
    z_dim,
    config=None,
):
    if model_type == "A7X":
        model = get_a7x(
            z_dim=z_dim,
            config=config,
        )
    elif model_type == "MNM":
        model = get_mnm(
            z_dim=z_dim,
            config=config,
        )
    elif model_type == "MCR":
        model = get_mcr(
            z_dim=z_dim,
            config=config,
        )
    elif model_type == "VB":
        model = get_vb(
            z_dim=z_dim,
            config=config,
        )
    elif model_type == "PR":
        return get_pr(z_dim=z_dim)
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")
    model.to(device)
    model.eval()
    return model


def get_mcr(
    z_dim,
    config=None,
):
    if config is None:
        config = CONFIG_MCR[z_dim]
        config["dimension"] = z_dim
    return MicroBERTCurveReducerModel(
        config=config,
    )


def get_pr(
    z_dim,
):
    return IncrementalPCA(
        n_components=z_dim,
    )


def get_mnm(
    z_dim,
    config=None,
):
    if config is None:
        config = CONFIG_MNM
        config["z_dim"] = z_dim
    return MLPNetworkModel(
        config=config,
    )


def get_a7x(
    z_dim,
    config=None,
):
    if config is None:
        config = CONFIG_A7X
        config["z_dim"] = z_dim
    return Autoencoder7XModel(
        config=config,
    )


def get_vb(
    z_dim,
    config=None,
):
    if config is None:
        config = CONFIG_VB
        config["reduction"] = 128 // z_dim
    return VAEBottleneckModel(
        config=config,
    )
