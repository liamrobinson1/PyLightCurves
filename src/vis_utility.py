import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Union


def new3figure() -> plt.figure:
    """Creates a new figure with 3D axes"""
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    return fig


def shg():
    """Shows current plots"""
    plt.show()


def scatter3(
    v: np.ndarray,
    cdata: Union[np.ndarray, str] = "blue",
    s: float = 1,
    marker: str = ",",
) -> plt.axis:
    """

    Args:
        v (np.ndarray nx3): Vector to scatter
        cdata (np.ndarray nx3 || str): Color RGB(A) per point or color string
        s (float): Marker size
        marker (str): Marker label

    Returns:
        plt.axis: Axes of produced scatter plot

    """
    assert 3 in v.shape, TypeError("scatter3 requires a 3xn or nx3 input vector")
    if v.shape[0] == 3:
        v = np.transpose(v)

    ax = plt.gca()
    ax.scatter3D(v[:, 0], v[:, 1], v[:, 2], c=cdata, s=s, marker=marker)
    return ax


def vec2cdata(v: np.ndarray, map_name: str):
    """Maps a 1D vector to a colormap
    Highest value mapped to end of colormap, lowest to beginning

    Args:
        v (np.ndarray nx1): Vector of floats
        map_name (str): Matplotlib colormap to use

    Returns:
        np.ndarray nx3: Colormap values at normalized indices

    """
    old_cmap = mpl.colormaps[map_name].resampled(256)
    uv = (v - np.min(v)) / (np.max(v) - np.min(v))
    return old_cmap(uv)


def axis(mode: str = "equal"):
    """Replicates axis(mode) functionality from MATLAB"""
    ax = plt.gca()
    ax.set_aspect(mode, "box")
