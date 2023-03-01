import numpy as np
from typing import Tuple, Callable, Any
from .math_utility import *
from .astro_const import AstroConstants


def rigid_rotation_dynamics(
    itensor: np.ndarray, w: np.ndarray, torque: np.ndarray = np.array([0, 0, 0])
) -> np.ndarray:
    dwdt = np.zeros((3,))
    (ix, iy, iz) = np.diag(itensor)
    (wx, wy, wz) = w
    (mx, my, mz) = torque

    dwdt[0] = -1 / ix * (iz - iy) * wy * wz + mx / ix
    dwdt[1] = -1 / iy * (ix - iz) * wz * wx + my / iy
    dwdt[2] = -1 / iz * (iy - ix) * wx * wy + mz / iz
    return dwdt


def gravity_gradient_torque(
    itensor: np.ndarray, rvec: np.ndarray, mu: float = AstroConstants.earth_mu
) -> np.ndarray:
    rmag = np.linalg.norm(rvec)
    # Distance from central body COM to satellite COM
    return 3 * mu / rmag**5 * np.cross(rvec, itensor @ rvec)
    # Torque applied to body


def quat_kinematics(q: np.ndarray, w: np.ndarray) -> np.ndarray:
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
    return 2 * np.arccos(dot(q1, q2))


def mrp_add(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    return quat_to_mrp(quat_add(mrp_to_quat(s1), mrp_to_quat(s2)))


def rv_add(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return quat_to_rv(quat_add(rv_to_quat(p1), rv_to_quat(p2)))


def quat_add(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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
    return quat_to_rv(mrp_to_quat(s))


def mrp_to_dcm(s: np.ndarray) -> np.ndarray:
    return quat_to_dcm(mrp_to_quat(s))


def mrp_to_quat(s: np.ndarray) -> np.ndarray:
    s2 = np.sum(s * s, axis=1, keepdims=True)
    return np.hstack((2 * s / (1 + s2), (1 - s2) / (1 + s2)))


def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    return q[:, 0:3] / (q[:, [3]] + 1)


def rv_to_dcm(p: np.ndarray) -> np.ndarray:
    return quat_to_dcm(rv_to_quat(p))


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
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
    theta = 2 * np.arccos(q[:, [3]])
    lam = hat(q[:, 0:3])
    lam[np.isnan(lam[:, 0]), :] = 0.0
    return theta * lam


def rv_to_quat(p: np.ndarray) -> np.ndarray:
    theta = vecnorm(p)
    lam = hat(p)
    q = np.hstack((lam * np.sin(theta / 2), np.cos(theta / 2)))
    return q


def axis_rotation_matrices() -> Tuple[Callable, Callable, Callable]:
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
    qinv = -q
    qinv[:, 3] = -qinv[:, 3]
    return qinv
