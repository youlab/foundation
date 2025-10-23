import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from sklearn.metrics import r2_score

from applications.utils.data import interpolate_y
from applications.utils.latents import reconstruct
from ml.utils.load_models import load_model


if __name__ == "__main__":

    model = load_model(
        model_name=model_name,
        model_type=model_type,
    )
    
    clip = np.load("src/clip.npy")
    length = clip.shape[1]
    interp_len = 16
    chunks = length // interp_len

    clip_interp = interpolate_y(y=clip.reshape(-1, interp_len))
    clip_interp_norm = clip_interp / clip_interp.max(axis=1).reshape(-1, 1)
    recon_interp = resample(
        (
            reconstruct(
                model=model,
                x=clip_interp_norm,
                batch_size=4096,
            ) * clip_interp.max(axis=1).reshape(-1, 1)
        ).reshape(-1, 128 * chunks),
        length,
        axis=1,
    )

    np.save("src/recon_clip2.npy", recon_interp)

    snippet = np.load("src/snippet.npy")
    snippet = snippet.reshape(-1, 128)
    snippet_norm = snippet / snippet.max(axis=1).reshape(-1, 1)
    recon_snippet = reconstruct(
        model=model,
        x=snippet_norm,
        batch_size=4096,
    ) * snippet.max(axis=1).reshape(-1, 1)

    np.save("src/recon_snippet.npy", recon_snippet.flatten())
