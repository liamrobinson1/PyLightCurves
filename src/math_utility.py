import numpy as np
from scipy.special import *


def sind(v: np.ndarray) -> np.ndarray:
    """Emulates MATLAB sind() functionality"""
    return np.sin(np.deg2rad(v))


def cosd(v: np.ndarray) -> np.ndarray:
    """Emulates MATLAB cosd() functionality"""
    return np.cos(np.deg2rad(v))


def tand(v: np.ndarray) -> np.ndarray:
    """Emulates MATLAB tand() functionality"""
    return np.tan(np.deg2rad(v))


def acosd(v: np.ndarray) -> np.ndarray:
    """Emulates MATLAB acosd() functionality"""
    return np.rad2deg(np.arccos(v))


def atand(v: np.ndarray) -> np.ndarray:
    """Emulates MATLAB atand() functionality"""
    return np.rad2deg(np.arctan(v))


def atan2d(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Emulates MATLAB atan2d() functionality"""
    return np.rad2deg(np.arctan2(y, x))


def hat(v: np.ndarray) -> np.ndarray:
    """Normalizes input np.ndarray nx3 such that each row is unit length
    If a row is [0,0,0], hat() returns it unchanged
    """
    vm = vecnorm(v)
    vm[vm == 0] = 1.0  # Such that hat([0,0,0]) = [0,0,0]
    return v / vm


def vecnorm(v: np.ndarray) -> np.ndarray:
    """Emulates MATLAB's vecnorm() functionality, returns L2 norm of rows"""
    axis = v.ndim - 1
    return np.linalg.norm(v, axis=axis, keepdims=True)


def dot(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns Euclidean dot product of rows of v1 and v2"""
    axis = v1.ndim - 1
    return np.sum(v1 * v2, axis=axis, keepdims=True)


def rdot(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Rectified dot product, where dot(v1,v2) < 0 -> 0"""
    d = dot(v1, v2)
    d[d < 0] = 0
    return d


def cart_to_sph(x, y, z):
    """Converts from Cartesian (x,y,z) to spherical (azimuth, elevation, range)"""
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph_to_cart(az, el, r):
    """Converts from spherical (azimuth, elevation, range) to Cartesian (x, y, z)"""
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def wrap_to_360(lon: np.array) -> np.array:
    """Replicates MATLAB's wrapTo3690() functionality"""
    lon %= 360
    lon[(lon == 0) & (lon > 0)] = 360
    return lon


def unique_rows(v: np.array, **kwargs) -> np.array:
    """

    Args:
         v (np.ndarray nx3): Input vector
         **kwargs: Additional arguments to pass to np.unique()

    Returns:
        np.ndarray mx3, m<=n: v with duplicate rows removed

    """
    return np.unique(np.round(v, decimals=6), axis=0, **kwargs)


def merge_clusters(v: np.ndarray, atol: float, miter: int = 3) -> np.ndarray:
    """Merges clusters of vectors within an angle tolerance by addition

    Args:
        v (np.ndarray nx3): Array of row vectors to merge
        atol (float) [rad]: Angle between vectors to merge by
        miter (int): Merge iterations

    Returns:
        np.ndarray mx3, m<=n: Merged vectors

    """
    n = v.shape[0]
    for i in range(miter):
        vh = hat(v)
        vm = np.empty((0, 3))
        merged_inds = np.array([])
        for i in range(n):
            ang_to_others = np.arccos(dot(np.tile(vh[i, :], (n, 1)), vh)).flatten()
            cluster_inds = np.argwhere(ang_to_others < atol).flatten()
            unmerged_cluster_inds = np.setdiff1d(cluster_inds, merged_inds)
            vm = np.append(vm, [np.sum(v[unmerged_cluster_inds, :], axis=0)], axis=0)
            merged_inds = np.append(merged_inds, unmerged_cluster_inds)
        v = vm

    zero_row_inds = np.argwhere((vecnorm(v) < 1e-12).flatten()).flatten()
    v = np.delete(v, zero_row_inds, axis=0)
    return unique_rows(v)


def points_to_planes(pt: np.array, plane_n: np.array, support: np.array) -> np.array:
    """Computes distance from a set of points to a set of planes

    Args:
        pt (np.ndarray nx3): Array of points in R^3
        plane_n (np.ndarray nx3): Normal vectors of planes
        support (np.ndarray nx1): Distance from each plane to the origin

    Returns:
        np.ndarray nx1: Distance from points to planes

    """
    return dot(pt, plane_n) + support


def close_egi(egi: np.array) -> np.array:
    """Enforces closure condition by adding mean closure error to each row

    Args:
        egi (np.ndarray nx3): Extended Gaussian Image

    Returns:
        np.ndarray nx3: EGI shifted such that the sum of rows is zero

    """
    return egi - np.sum(egi, axis=0) / egi.shape[0]


def remove_zero_rows(v: np.array) -> np.array:
    """Removes any rows from v (np.ndarray nx3) that are all zeros"""
    return np.delete(v, vecnorm(v).flatten() == 0, axis=0)


def elliptic_pi_complete(n: np.ndarray, ksquared: np.ndarray) -> np.ndarray:
    return elliprf(0, 1 - ksquared, 1) + 1 / 3 * n * elliprj(0, 1 - ksquared, 1, 1 - n)


def elliptic_pi_incomplete(n: np.ndarray, phi: np.ndarray, ksquared: np.ndarray):
    c = np.floor((phi + np.pi / 2) / np.pi)
    phi_shifted = phi - c * np.pi
    onemk2_sin2phi = 1 - ksquared * np.sin(phi_shifted) ** 2
    cos2phi = np.cos(phi_shifted) ** 2
    sin3phi = np.sin(phi_shifted) ** 3
    n_sin2phi = n * np.sin(phi_shifted) ** 2

    periodic_portion = 2 * c * elliptic_pi_complete(n, ksquared)

    return (
        np.sin(phi_shifted) * elliprf(cos2phi, onemk2_sin2phi, 1)
        + 1 / 3 * n * sin3phi * elliprj(cos2phi, onemk2_sin2phi, 1, 1 - n_sin2phi)
        + periodic_portion
    )


def stack_mat_mult(mats, v) -> np.ndarray:
    """Multiplies each row in v by each page of mats

    Args:
        mats (np.ndarray m x m x n): Matrices to multiply by
        v (np.ndarray n x m): Vectors to be multiplied

    Returns:
        np.ndarray n x m: Multiplied product
    """
    (n, m) = v.shape

    if mats.shape[0] == mats.shape[1]:
        mats = np.moveaxis(mats, -1, 0)

    print(mats.shape, v.shape)
    assert mats.shape[0] == n, "Matrix dimensions to not match vector!"
    v_deep = np.reshape(v, (n, m, 1))
    return np.squeeze(mats @ v_deep)
