import numpy as np
from typing import Union
import pyvista as pv
from .astro_const import AstroConstants
from .math_utility import hat


def plot_earth(pl: pv.Plotter):
    sphere = pv.Sphere(
        radius=AstroConstants.earth_r_eq,
        theta_resolution=120,
        phi_resolution=120,
        start_theta=270.001,
        end_theta=270,
    )
    sph_hat_pts = hat(sphere.points)
    sphere.active_t_coords = np.zeros((sphere.points.shape[0], 2))
    sphere.active_t_coords[:, 0] = 0.5 + np.arctan2(
        -sph_hat_pts[:, 0], sph_hat_pts[:, 1]
    ) / (2 * np.pi)
    sphere.active_t_coords[:, 1] = 0.5 + np.arcsin(sph_hat_pts[:, 2]) / np.pi

    earth = pv.Texture("resources/textures/earth_tex.jpg")
    pl.add_mesh(sphere, texture=earth, smooth_shading=False)


def scatter3(
    pl: pv.Plotter,
    v: np.ndarray,
    **kwargs
):
    """

    Args:
        v (np.ndarray nx3): Vector to scatter

    Returns:
        

    """
    assert 3 in v.shape, TypeError("scatter3 requires a 3xn or nx3 input vector")
    if v.shape[0] == 3:
        v = np.transpose(v)

    pc = pv.PolyData(v)
    pl.add_mesh(pc, 
                render_points_as_spheres=True,
                **kwargs)
