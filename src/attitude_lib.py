import numpy as np
import scipy
from typing import Tuple, Callable, Any
from .math_utility import *
from .astro_const import AstroConstants

def integrate_rigid_attitude_dynamics(q0: np.ndarray,
                                      omega0: np.ndarray, 
                                      itensor: np.ndarray,
                                      body_torque: Callable,
                                      teval: np.ndarray,
                                      int_tol=1e-13) -> np.ndarray:
    """Integration for rigid body rotational dynamics

    Args:
        q0 (np.ndarray 1x3): Initial quaternion from inertial to body frame
        omega0 (np.ndarray 1x3) [rad/s]: Angular velocity vector of body relative to inertial space
        itensor (np.ndarray 3x3) [kg m^2]: Inertia tensor in principal axes, should be diagonal
        body_torque (Callable -> np.ndarray 1x3) [Nm]: Torque applied to the body due to external forces
        teval (np.ndarray 1xn) [seconds]: Times to return integrated trajectory at
        int_tol (float): Integration rtol and atols for RK45

    Returns:
        np.ndarray nx7: Integrated quaternion [:4, :] and angular velocity [4:, :]

    """
    fun = lambda t, y: np.concatenate(
        (quat_kinematics(y[:4], y[4:]), rigid_rotation_dynamics(t, y[4:], itensor, body_torque))
    )
    tspan = [np.min(teval), np.max(teval)]
    y0 = np.concatenate((q0, omega0))
    ode_res = scipy.integrate.solve_ivp(fun, tspan, y0, t_eval=teval, rtol=int_tol, atol=int_tol)
    return ode_res.y.T

def rigid_rotation_dynamics(
    t: float, w: np.ndarray, itensor: np.ndarray, torque: Callable = None
) -> np.ndarray:
    """Rigid body rotational dynamics (Euler's EOMs)

    Args:
        itensor (np.ndarray 3x3) [kg m^2]: Inertia tensor in principal axes, should be diagonal
        t (float) [seconds]: Current integration time
        w (np.ndarray 1x3) [rad/s]: Angular velocity vector of body relative to inertial space
        torque (Callable(t,y) -> np.ndarray 1x3) [Nm]: Torque applied to the body due to external forces

    Returns:
        np.ndarray: Time derivative of angular velocity vector

    """
    dwdt = np.zeros((3,))
    (ix, iy, iz) = np.diag(itensor)
    (wx, wy, wz) = w
    if torque is not None:
        (mx, my, mz) = torque(t, w)
    else:
        (mx, my, mz) = (0, 0, 0)

    dwdt[0] = -1 / ix * (iz - iy) * wy * wz + mx / ix
    dwdt[1] = -1 / iy * (ix - iz) * wz * wx + my / iy
    dwdt[2] = -1 / iz * (iy - ix) * wx * wy + mz / iz
    return dwdt


def gravity_gradient_torque(
    itensor: np.ndarray, rvec: np.ndarray, mu: float = AstroConstants.earth_mu
) -> np.ndarray:
    """Computes gravity gradient torque in the body frame

    Args:
        itensor (np.ndarray 3x3) [kg m^2]: Inertia tensor in body axes
        rvec (np.ndarray 1x3) [km]: Position vector of body
        mu (float) [km^3/s^2]: Gravitational parameter of central body

    Returns:
        np.ndarray 1x3 [Nm]: Gravity gradient torque [Nm] in body frame

    """
    rmag = np.linalg.norm(rvec)
    # Distance from central body COM to satellite COM
    return 3 * mu / rmag**5 * np.cross(rvec, itensor / 1e6 @ rvec) * 1e3
    # Torque applied to body [Nm]


def quat_kinematics(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Kinematic differential equations for quaternion time evolution

    Args:
        q (np.ndarray 1x4): Current quaternion from inertial to body
        w (np.ndarray 1x3) [rad/s]: Current angular velocity in body frame

    Returns:
        np.ndarray 1x3: Time derivative of input quaternion

    """
    return (
        1
        / 2
        * np.array(
            [
                [q[3], -q[2], q[1], q[0]],
                [q[2], q[3], -q[0], q[1]],
                [-q[1], q[0], q[3], q[2]],
                [-q[0], -q[1], -q[2], q[3]],
            ]
        )
        @ np.concatenate((w, [0]))
    )


def quat_ang(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle between two quaternion arrays

    Args:
        q1 (np.ndarray nx4): Quaternion array
        q2 (np.ndarray nx4): Quaternion array

    Returns:
        np.ndarray nx1 [rad]: Angle between quaternion arrays

    """
    return 2 * np.arccos(dot(q1, q2))


def mrp_add(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Adds modified Rodrigues parameters (see docstring for quat_add)

    Args:
        s1 (np.ndarray nx3): modified Rodrigues parameter array
        s2 (np.ndarray nx3): modified Rodrigues parameter array

    Returns:
        np.ndarray nx3: modified Rodrigues parameter array

    """
    return quat_to_mrp(quat_add(mrp_to_quat(s1), mrp_to_quat(s2)))


def rv_add(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Adds rotation vectors (see docstring for quat_add)

    Args:
        p1 (np.ndarray nx3): Rotation vector array
        p2 (np.ndarray nx3): Rotation vector array

    Returns:
        np.ndarray nx3: Rotation vector array

    """
    return quat_to_rv(quat_add(rv_to_quat(p1), rv_to_quat(p2)))


def quat_add(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Adds (multiplies) quaternions together such that
    quat_add(inv_quat(q1), q2) gives the rotation between q1 and q2

    Args:
        q1 (np.ndarray nx4): Quaternion array
        q2 (np.ndarray nx4): Quaternion array

    Returns:
        np.ndarray nx4: Quaternion array

    """
    nquats = q1.shape[0]
    q1d = np.reshape(q1, (nquats, 4, 1))
    q2dmat = np.array(
        [
            [q2[:, 3], q2[:, 2], -q2[:, 1], q2[:, 0]],
            [-q2[:, 2], q2[:, 3], q2[:, 0], q2[:, 1]],
            [q2[:, 1], -q2[:, 0], q2[:, 3], q2[:, 2]],
            [-q2[:, 0], -q2[:, 1], -q2[:, 2], q2[:, 3]],
        ]
    )
    q12 = np.matmul(np.moveaxis(q2dmat, 2, 0), q1d)
    return np.reshape(q12, (nquats, 4))


def mrp_to_rv(s: np.ndarray) -> np.ndarray:
    """Converts modified Rodrigues parameters to rotation vectors

    Args:
        s (np.ndarray nx3): Modified Rodrigues parameter array

    Returns:
        np.ndarray nx3: Rotation vector array

    """
    return quat_to_rv(mrp_to_quat(s))


def mrp_to_dcm(s: np.ndarray) -> np.ndarray:
    """Converts modified Rodrigues parameters to direction cosine matrices

    Args:
        s (np.ndarray nx3): Modified Rodrigues parameter array

    Returns:
        np.ndarray nx3x3: Direction cosine matrix array

    """
    return quat_to_dcm(mrp_to_quat(s))


def mrp_to_quat(s: np.ndarray) -> np.ndarray:
    """Converts modified Rodrigues parameters to quaternions

    Args:
        s (np.ndarray nx3): Modified Rodrigues parameter array

    Returns:
        (np.ndarray nx4): Quaternion array

    """
    s2 = np.sum(s * s, axis=1, keepdims=True)
    return np.hstack((2 * s / (1 + s2), (1 - s2) / (1 + s2)))


def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    """Converts quaternions to modified Rodrigues parameters

    Args:
        q (np.ndarray nx4): Quaternion array

    Returns:
        np.ndarray nx3: modified Rodrigues parameter array

    """
    return q[:, 0:3] / (q[:, [3]] + 1)


def rv_to_dcm(p: np.ndarray) -> np.ndarray:
    """Converts rotation vectors to direction cosine matrices

    Args:
        p (np.ndarray nx3): Rotation vector array

    Returns:
        np.ndarray nx3x3: Direction cosine matrix array

    """
    return quat_to_dcm(rv_to_quat(p))


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Converts quaternions to direction cosine matrices

    Args:
        q (np.ndarray nx4): Quaternion array

    Returns:
        np.ndarray nx3x3: Direction cosine matrix array

    """
    if q.ndim == 1:
        q = np.reshape(q, (1, 4))
    C = np.empty((3, 3, q.shape[0]))
    C[0, 0, :] = 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 2] ** 2
    C[0, 1, :] = 2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
    C[0, 2, :] = 2 * (q[:, 0] * q[:, 2] - q[:, 1] * q[:, 3])
    C[1, 0, :] = 2 * (q[:, 0] * q[:, 1] - q[:, 2] * q[:, 3])
    C[1, 1, :] = 1 - 2 * q[:, 0] ** 2 - 2 * q[:, 2] ** 2
    C[1, 2, :] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    C[2, 0, :] = 2 * (q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3])
    C[2, 1, :] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    C[2, 2, :] = 1 - 2 * q[:, 0] ** 2 - 2 * q[:, 1] ** 2
    return np.squeeze(C)


def quat_to_rv(q: np.ndarray) -> np.ndarray:
    """Converts quaternions to rotation vectors

    Args:
        q (np.ndarrah nx4): Quaternion array

    Returns:
        np.ndarray nx3: Rotation vector array

    """
    theta = 2 * np.arccos(q[:, [3]])
    lam = hat(q[:, 0:3])
    lam[np.isnan(lam[:, 0]), :] = 0.0
    return theta * lam


def rv_to_quat(p: np.ndarray) -> np.ndarray:
    """Converts rotation vectors to quaternions

    Args:
        p (np.ndarray nx3): Rotation vector array

    Returns:
        np.ndarray nx3: Quaternion array

    """
    theta = vecnorm(p)
    lam = hat(p)
    q = np.hstack((lam * np.sin(theta / 2), np.cos(theta / 2)))
    return q


def axis_rotation_matrices() -> Tuple[Callable, Callable, Callable]:
    """Returns rotation matrices about the three body axes

    Args:


    Returns:
        tuple: callables for body axis rotations

    """
    r1 = lambda t: np.array(
        [[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]]
    )

    r2 = lambda t: np.array(
        [[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]]
    )

    r3 = lambda t: np.array(
        [[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]]
    )

    return (r1, r2, r3)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Finds the quaternion inverse (conjugate for unit length quaternions)

    Args:
        q (np.ndarray nx4): Input quaternion array

    Returns:
        np.ndarray nx3: Inverse quaternion array

    """
    qinv = -q
    qinv[:, 3] = -qinv[:, 3]
    return qinv
