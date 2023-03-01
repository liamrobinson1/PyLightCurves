import numpy as np

def sind(v: np.ndarray) -> np.ndarray:
    return np.sin(np.deg2rad(v))

def cosd(v: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(v))

def tand(v: np.ndarray) -> np.ndarray:
    return np.tan(np.deg2rad(v))

def acosd(v: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(v))

def atand(v: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arctan(v))

def hat(v: np.ndarray) -> np.ndarray:
    vm = vecnorm(v)
    vm[vm==0] = 1.0 # Such that hat([0,0,0]) = [0,0,0]
    return v / vm

def vecnorm(v: np.ndarray) -> np.ndarray:
    axis = v.ndim - 1
    return np.linalg.norm(v, axis=axis, keepdims=True)

def dot(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    axis = v1.ndim - 1
    return np.sum(v1 * v2, axis=axis, keepdims=True)

def rdot(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    d = dot(v1, v2)
    d[d<0] = 0
    return d

def cart_to_sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def sph_to_cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def wrap_to_360(lon: np.array) -> np.array:
    lon %= 360
    lon[(lon == 0) & (lon > 0)] = 360
    return lon

def unique_rows(v: np.array, **kwargs) -> np.array:
    return np.unique(np.round(v, decimals=6), axis=0, **kwargs)

def merge_clusters(v: np.array, atol: float, miter: int = 3) -> np.array:
    n = v.shape[0]
    for i in range(miter):
        vh = hat(v)
        vm = np.empty((0,3))
        merged_inds = np.array([])
        for i in range(n):
            ang_to_others = np.arccos(dot(np.tile(vh[i,:], (n,1)), vh)).flatten()
            cluster_inds = np.argwhere(ang_to_others < atol).flatten()
            unmerged_cluster_inds = np.setdiff1d(cluster_inds, merged_inds)
            vm = np.append(vm, [np.sum(v[unmerged_cluster_inds,:], axis=0)], axis=0)
            merged_inds = np.append(merged_inds, unmerged_cluster_inds)
        v = vm
    
    zero_row_inds = np.argwhere((vecnorm(v)<1e-12).flatten()).flatten()
    v = np.delete(v, zero_row_inds, axis=0) 
    return unique_rows(v)

def points_to_planes(pt: np.array, plane_n: np.array, support: np.array) -> np.array:
    return dot(pt, plane_n) + support

def close_egi(egi: np.array) -> np.array:
    return egi - np.sum(egi, axis=0) / egi.shape[0]

def remove_zero_rows(v: np.array) -> np.array:
    return np.delete(v, vecnorm(v).flatten() == 0, axis=0)