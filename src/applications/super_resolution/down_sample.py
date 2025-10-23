import random

import numpy as np
from scipy.interpolate import interp1d


def interpolate_y(
    x,
    y,
    interp_len,    
):
    f = interp1d(
        x=x,
        y=y,
        assume_sorted=True,
    )
    return f(
        np.linspace(
            x.min(),
            x.max(),
            interp_len,
        )
    )


def get_down_sample(
    x: np.ndarray,
    mask_pct: float,
    use_fixed_interval: bool=True,
):
    """
    This function returns a down-sampled array from the input array to be used in the super 
    resolution tasks. The random intervals are created using a random seed, so repeated calls
    should return the same values.

    PARAMETERS
    ----------
    x -> np.ndarray: this is the input array of shape (n_samples, seq_len)
    mask_pct -> float: this is the percentage of the input to mask
    use_fixed_interval -> bool: if True, then the samples are taken in a fixed interval, otherwise
    the first and last value are kept, and the others are returned randomly.
    """
    if (
        mask_pct < 2
    ) or (
        mask_pct > 97
    ):
        raise ValueError(f"Variable 'mask_pct' should be between 2 and 97, inclusive, but is {mask_pct}")

    seq_len = x.shape[1]
    n_samples = int(seq_len * (100 - mask_pct) / 100)

    if use_fixed_interval:
        idx_downsample = np.linspace(
            start=0,
            stop=seq_len - 1,
            num=n_samples,
            dtype=int,
        )
    else:
        idx_downsample = [0]
        idx_downsample.extend(
            list(
                sorted(
                    random.Random(42).sample(
                        np.linspace(
                            start=1,
                            stop=seq_len - 2,
                            num=seq_len - 2,
                            dtype=int,
                        ).tolist(),
                        n_samples - 2,
                    )
                )
            )
        )
        idx_downsample.append(seq_len - 1)
        idx_downsample = np.array(idx_downsample)

    return (
        x[:, idx_downsample],
        interpolate_y(
            x=idx_downsample,
            y=x[:, idx_downsample],
            interp_len=seq_len,
        ),
    )
