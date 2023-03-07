import numpy as np
import os
from typing import Union
import pyvista as pv
from .astro_const import AstroConstants
from .math_utility import hat, vecnorm
from .object_lib import Object
from .attitude_lib import quat_to_rv


def vis_attitude_motion(
    obj: Object,
    quat: np.ndarray,
    fname: str = "test.mp4",
    framerate: int = 60,
    quality: int = 9,
    background_color: str = "white",
):
    pl = pv.Plotter()
    pl.open_movie(f"out/{fname}", framerate=framerate, quality=quality)
    pl.set_background(background_color)
    pl.add_mesh(obj._mesh)
    o_obj = obj._mesh.copy()
    for i in range(quat.shape[0]):
        rv = quat_to_rv(quat[i, :])
        new_obj = obj._mesh.copy().rotate_vector(
            vector=hat(rv), angle=np.rad2deg(vecnorm(rv))
        )
        obj._mesh.copy_from(new_obj)
        pl.write_frame()
        obj._mesh.copy_from(o_obj)
    pl.close()


def two_sphere(pl: pv.Plotter, radius: float = 1.0, opacity: float = 0.3, **kwargs):
    sphere = pv.Sphere(
        radius=radius,
        theta_resolution=120,
        phi_resolution=120,
        start_theta=270.001,
        end_theta=270,
    )
    pl.add_mesh(sphere, opacity=opacity, **kwargs)


def plot_earth(pl: pv.Plotter):
    sphere = pv.Sphere(
        radius=AstroConstants.earth_r_eq,
        theta_resolution=120,
        phi_resolution=120,
        start_theta=270.001,
        end_theta=270,
    )
    contours = sphere.contour(scalars=sphere.points[:, 2], isosurfaces=20)
    sph_hat_pts = hat(sphere.points)
    sphere.active_t_coords = np.zeros((sphere.points.shape[0], 2))
    sphere.active_t_coords[:, 0] = 0.5 + np.arctan2(
        -sph_hat_pts[:, 0], sph_hat_pts[:, 1]
    ) / (2 * np.pi)
    sphere.active_t_coords[:, 1] = 0.5 + np.arcsin(sph_hat_pts[:, 2]) / np.pi

    earth = pv.Texture("resources/textures/earth_tex.jpg")
    pl.add_mesh(sphere, texture=earth, smooth_shading=False)


def scatter3(pl: pv.Plotter, v: np.ndarray, **kwargs):
    """Replicates MATLAB scatter3() with pyvista backend

    Args:
        v (np.ndarray nx3): Vector to scatter

    Returns:


    """
    assert 3 in v.shape, TypeError("scatter3 requires a 3xn or nx3 input vector")
    if v.shape[0] == 3:
        v = np.transpose(v)

    pc = pv.PolyData(v)
    pl.add_mesh(pc, render_points_as_spheres=True, **kwargs)


def plot3(pl: pv.Plotter, v: np.ndarray, **kwargs) -> pv.PolyData:
    """Replicates MATLAB plot3() with pyvista backend
    Use densely scattered points to avoid confusing splines

    Args:
        v (np.ndarray nx3): Vector to plot

    Returns:


    """
    assert 3 in v.shape, TypeError("plot3 requires a 3xn or nx3 input vector")
    if v.shape[0] == 3:
        v = np.transpose(v)

    spline = pv.Spline(v, v.shape[0])
    pl.add_mesh(spline, render_lines_as_tubes=True, **kwargs)
    return spline


def texit(ch: pv.Chart2D, title: str, xlabel: str, ylabel: str, zlabel: str = ""):
    """All my prefered plot formatting, all in one place

    Args:
        ch (pyvista.Chart2D): Chart to format
        title (str): Title string
        xlabel (str): String for the x-axis
        ylabel (str): String for the y-axis
        zlabel (str | None): String for the z-axis

    Returns:


    """
    label_size = 25
    ch.title = title
    ch.x_label = xlabel
    ch.y_label = ylabel
    ch.x_axis.label_size = label_size
    ch.y_axis.label_size = label_size
    ch.x_axis.tick_label_size = label_size
    ch.y_axis.tick_label_size = label_size
    t = ch.GetTitleProperties()
    t.SetBold(True)
    t.SetFontSize(30)
    ch.background_color = "white"


def show_and_copy(ch: pv.Chart2D):
    im_arr = ch.show(screenshot=True, off_screen=True)
    from PIL import Image

    im = Image.fromarray(im_arr)
    imf = "imtemp.png"
    im.save(imf)

    cp_cmd = (
        f"osascript -e 'on run argv'"
        f" -e 'set the clipboard to "
        f"(read POSIX file (POSIX path of first "
        f"item of argv) as JPEG picture)' "
        f"-e 'end run' {os.getcwd()}/{imf}"
    )

    os.system(cp_cmd)
    os.remove(imf)
    ch.show()
