import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm import trange

from config import (
    SEQ_LEN,
    Z_DIM,
)
from ml.models.vae_bottleneck import VAEBottleneckModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_latents(
    x,
    model=None,
    use_transformer=False,
):
    """
    Take an input array x and calculate the latent values.

    PARAMETERS
    ----------
    x -> numpy.array : input vector of shape (N, 1, SEQ_LEN)

    RETURNS
    -------
    mu -> numpy.array : latent representation of x
    """
    if model is None:
        raise ValueError(f"model must be defined but is None")

    if x.shape[1] != 1:
        raise ValueError(f"x.shape[1] should be 1 but it is {x.shape[1]}")
    if x.shape[2] != SEQ_LEN:
        raise ValueError(f"x.shape[2] should be {SEQ_LEN} but it is {x.shape[2]}")
    
    with torch.no_grad():
        if use_transformer:
            mu = model.get_embedding(x)
        else:
            if isinstance(model, VAEBottleneckModel):
                encoded = model.encoder(x).reshape(x.shape[0], SEQ_LEN, -1)
                mu = model.mu(encoded)
                mu = mu.reshape(x.shape[0], -1)
            else:
                encoded = model.encoder(x).reshape(x.shape[0], -1)
                mu = model.mu(encoded)
    return mu


def get_latents(
    z,
    batch_size=4_096,
    use_transformer=False,
    z_dim=Z_DIM,
    model=None,
):
    """
    This get_latents function takes an array z of shape [n_batch, SEQ_LEN] and converts it to the
    latent representation using the VAE.
    
    PARAMETERS
    ----------
    z -> numpy.array : input array
    batch_size -> int : batch size for encoding
    use_transformer -> bool : whether or not to use transformer model
    z_dim -> int : number of latent dimensions
    model -> torch model : optional to load in model
    
    RETURNS
    -------
    out -> numpy.array : latent vector of shape [n_batch, Z_DIM + 1], where the last column is the
        max value from the input
    """
    z_max = z.max(axis=1)
    x = np.zeros_like(z)
    x[z_max > 0, :] = z[z_max > 0, :] / z_max[z_max > 0].reshape(-1, 1)
    x[z_max == 0, :] = 1

    if model is None:
        raise ValueError(f"model must be defined but is None")

    elif isinstance(model, IncrementalPCA,):
        out = np.zeros((z.shape[0], z_dim + 1,),)
        out[:, -1] = z.max(axis=1)
        out[:, :-1] = model.transform(x)
        return out
    
    model.eval()
    x = torch.tensor(x).to(device).float()
    out = torch.zeros((z.shape[0], z_dim + 1,),).to(device)
    batches = z.shape[0] // batch_size + (0 if z.shape[0] % batch_size == 0 else 1)
    for batch in range(batches):
        i1 = batch * batch_size
        i2 = i1 + batch_size
        out[i1:i2, :-1] = calc_latents(
            x=x[i1:i2].reshape(-1, 1, SEQ_LEN,),
            use_transformer=use_transformer,
            model=model,
        )
    
    out = out.cpu().detach().numpy()
    out[:, -1] = z.max(axis=1)
    return out


def decode(
    z,
    model=None,
    batch_size=4096,
):
    """
    This decode function takes a latent array z of shape [n_batch, 1, Z_DIM + 1] where the first 
    Z_DIM values are the latent vector and the last value is the maximum y value.
    
    PARAMETERS
    ----------
    z -> numpy.array : latent vector including y_max
    batch_size -> int : batch size for decoding
    
    RETURNS
    -------
    out -> numpy.array : reconstructed curves of shape [n_batch, SEQ_LEN]
    """
    if model is None:
        raise ValueError(f"model must be defined but is None")

    batches = z.shape[0] // batch_size + (0 if z.shape[0] % batch_size == 0 else 1)
    out = np.zeros((z.shape[0], SEQ_LEN,),)
    for batch in trange(batches):
        i1 = batch * batch_size
        i2 = i1 + batch_size
        with torch.no_grad():
            out[i1:i2] = model.decoder(torch.tensor(z[i1:i2, :, :-1]).float().to(device)).cpu().detach().numpy()[:, 0, :]
    
    out = out * z[:, 0, -1:]
    return out


def reconstruct(
    model,
    x,
    batch_size,
    is_vae=True,
    is_pca=False,
):
    n = x.shape[0]
    recon = np.zeros_like(x)
    batches = n // batch_size + (0 if (n % batch_size == 0) else 1)
    if is_pca:
        for batch in range(batches):
            i1 = batch * batch_size
            i2 = i1 + batch_size
            latent_vectors = model.transform(x[i1:i2])
            recon[i1:i2] = model.inverse_transform(latent_vectors)
    else:
        model.eval()
        x = torch.tensor(x).view(n, 1, 128,).to(device).float()
        for batch in range(batches):
            i1 = batch * batch_size
            i2 = i1 + batch_size
            with torch.no_grad():
                if is_vae:
                    out, _, _ = model(x[i1:i2])
                else:
                    _, out = model(x[i1:i2])
                recon[i1:i2] = out[:, 0, :].cpu().detach().numpy()
    return recon
