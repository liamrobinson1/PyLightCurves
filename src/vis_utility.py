import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Union

def new3figure() -> plt.figure:
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    return fig

def shg():
    plt.show()

def scatter3(v: np.ndarray, cdata: Union[np.ndarray, str] = 'blue', 
             s: float = 1, marker: str = ',') -> plt.axis:
    assert 3 in v.shape, \
        TypeError("scatter3 requires a 3xn or nx3 input vector")
    if v.shape[0] == 3: v = np.transpose(v)

    ax = plt.gca()
    ax.scatter3D(v[:,0], v[:,1], v[:,2], 
                c=cdata, s=s, marker=marker)
    return ax

def vec2cdata(v: np.ndarray, map_name: str):
    old_cmap = mpl.colormaps[map_name].resampled(256)
    uv = (v - np.min(v)) / (np.max(v) - np.min(v))
    return old_cmap(uv);

def axis(mode: str = 'equal'):
    ax = plt.gca()
    ax.set_aspect(mode, 'box')
