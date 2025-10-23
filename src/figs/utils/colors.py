import numpy as np
from matplotlib import colormaps as cm


def get_colors(
    n,
    cmap_name="gist_earth",
):
    colormap = cm.get_cmap(cmap_name)
    return colormap(np.linspace(0.0, 0.9, n))
