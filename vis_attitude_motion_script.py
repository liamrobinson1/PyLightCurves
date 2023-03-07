from src.object_lib import Object
from src.attitude_lib import *
from src.vis_utility import *
import numpy as np


def vis_motion_and_poinsot(
    obj: Object,
    quat: np.ndarray,
    omega: np.ndarray,
    itensor: np.ndarray,
    fname: str = "test.mp4",
    framerate: int = 60,
    quality: int = 9,
    background_color: str = "white",
):
    angular_momentum = (itensor.T @ omega.T).T
    T = 1 / 2 * omega[0, :] @ itensor @ omega[0, :].T
    pl = pv.Plotter(shape=(1, 3), window_size=(2000, 2000 // 3))
    pl.open_movie(f"out/{fname}", framerate=framerate, quality=quality)
    pl.set_background(background_color)
    pl.subplot(0, 0)
    pl.add_text("Orientation", font_size=24, color="black")
    pl.add_mesh(obj._mesh)
    pl.subplot(0, 1)
    pl.add_text("Angular Momentum", font_size=24, color="black")
    sc = np.zeros(quat.shape[0])
    two_sphere(pl, np.mean(vecnorm(angular_momentum)), opacity=1, color="grey")
    kinetic_energy_ellipsoid = pv.ParametricEllipsoid(
        np.sqrt(2 * T * itensor[0, 0]),
        np.sqrt(2 * T * itensor[1, 1]),
        np.sqrt(2 * T * itensor[2, 2]),
    )
    pl.add_mesh(kinetic_energy_ellipsoid, opacity=1, color="purple")
    h_line = plot3(
        pl,
        angular_momentum,
        line_width=10,
        scalars=sc,
        cmap="plasma",
        show_scalar_bar=False,
    )
    pl.subplot(0, 2)
    pl.add_text("Rotation Vector", font_size=24, color="black")
    two_sphere(pl, np.pi + 0.2)
    rv_line = plot3(
        pl,
        quat_to_rv(quat),
        line_width=10,
        scalars=sc,
        cmap="plasma",
        show_scalar_bar=False,
    )

    o_obj = obj._mesh.copy()
    for i in range(quat.shape[0]):
        rv_sc = np.zeros(quat.shape[0])
        rv_sc[i - 1 : i + 1] = 1
        pl.update_scalars(rv_sc, mesh=rv_line, render=False)
        pl.update_scalars(rv_sc, mesh=h_line, render=False)

        rv = quat_to_rv(quat[i, :])
        new_obj = obj._mesh.copy().rotate_vector(
            vector=hat(rv), angle=np.rad2deg(vecnorm(rv))
        )
        obj._mesh.copy_from(new_obj)
        pl.write_frame()
        obj._mesh.copy_from(o_obj)
    pl.close()


obj = Object("tess.obj")

torques = lambda t, y: np.zeros(3)
itensor = np.diag([1, 2, 2])
q0 = np.array([0, 0, 0, 1])
w0 = np.array([1, 1, 3])
teval = np.linspace(0, 50, int(1e3))

res = integrate_rigid_attitude_dynamics(q0, w0, itensor, torques, teval)
# vis_motion_and_poinsot(obj, quat_upper_hemisphere(res[:, :4]), res[:, 4:], itensor)
vis_attitude_motion(obj, res[:, :4])
