import numpy as np
import scipy
from typing import Tuple, Callable, Any
from .math_utility import *
from .astro_const import AstroConstants


def integrate_rigid_attitude_dynamics(
    q0: np.ndarray,
    omega0: np.ndarray,
    itensor: np.ndarray,
    teval: np.ndarray,
    body_torque: Callable = lambda t, y: np.zeros(3),
    int_tol=1e-13,
) -> np.ndarray:
    """Integration for rigid body rotational dynamics

    Args:
        q0 (np.ndarray 1x3): Initial quaternion from inertial to body frame
        omega0 (np.ndarray 1x3) [rad/s]: Angular velocity vector of body relative to inertial space
        itensor (np.ndarray 3x3) [kg m^2]: Inertia tensor in principal axes, should be diagonal
        body_torque (Callable -> np.ndarray 1x3) [Nm]: Torque applied to the body due to external forces
        teval (np.ndarray 1xn) [seconds]: Times to return integrated trajectory at
        int_tol (float): Integration rtol and atols for RK45

    Returns:
        np.ndarray nx4: Integrated quaternion
        np.ndarray nx3: Integrated angular velocity [rad/s]

    """
    fun = lambda t, y: np.concatenate(
        (
            quat_kinematics(y[:4], y[4:]),
            rigid_rotation_dynamics(
                t, y[4:], itensor, lambda t: body_torque(t, y[0:4])
            ),
        )
    )
    tspan = [np.min(teval), np.max(teval)]
    y0 = np.concatenate((q0, omega0))
    ode_res = scipy.integrate.solve_ivp(
        fun, tspan, y0, t_eval=teval, rtol=int_tol, atol=int_tol
    )
    return (ode_res.y.T[:, :4], ode_res.y.T[::, 4:])


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
        (mx, my, mz) = torque(t)
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
    return (
        3 * mu / rmag**5 * np.cross(rvec.flatten(), (itensor @ rvec).flatten()) / 1e3
    )
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
    if q1.size == 4 and q2.shape[0] > 1:
        q1 = np.tile(q1, (q2.shape[0], 1))
    if q1.shape[0] > 1 and q2.size == 4:
        q2 = np.tile(q2, (q1.shape[0], 1))

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
        np.ndarray 3x3xn: Direction cosine matrix array

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
        np.ndarray 3x3xn: Direction cosine matrix array

    """
    return quat_to_dcm(rv_to_quat(p))


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Converts quaternions to direction cosine matrices

    Args:
        q (np.ndarray nx4): Quaternion array

    Returns:
        np.ndarray 3x3xn: Direction cosine matrix array

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
    if q.size == 4:
        q = np.reshape(q, (1, 4))
    theta = 2 * np.arccos(q[:, [3]])
    lam = hat(q[:, 0:3])
    lam[np.isnan(lam[:, 0]), :] = 0.0
    if q.size == 4:
        return (theta * lam).flatten()
    else:
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
        [
            [np.ones_like(t), np.zeros_like(t), np.zeros_like(t)],
            [np.zeros_like(t), np.cos(t), np.sin(t)],
            [np.zeros_like(t), -np.sin(t), np.cos(t)],
        ]
    )

    r2 = lambda t: np.array(
        [
            [np.cos(t), np.zeros_like(t), -np.sin(t)],
            [np.zeros_like(t), np.ones_like(t), np.zeros_like(t)],
            [np.sin(t), np.zeros_like(t), np.cos(t)],
        ]
    )

    r3 = lambda t: np.array(
        [
            [np.cos(t), np.sin(t), np.zeros_like(t)],
            [-np.sin(t), np.cos(t), np.zeros_like(t)],
            [np.zeros_like(t), np.zeros_like(t), np.ones_like(t)],
        ]
    )

    return (r1, r2, r3)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Finds the quaternion inverse (conjugate for unit length quaternions)

    Args:
        q (np.ndarray nx4): Input quaternion array

    Returns:
        np.ndarray nx3: Inverse quaternion array

    """
    if q.size == 4:
        q = np.reshape(q, (1, 4))
    qinv = -q
    qinv[:, 3] = -qinv[:, 3]
    return qinv


def quat_upper_hemisphere(q: np.ndarray) -> np.ndarray:
    """Transforms any quaternions in q to the upper hemisphere of S^3 (q[:,3] > 0)

    Args:
        q (np.ndarray nx4): Quaternion array

    Returns:
        np.ndarray nx3: Transformed quaternion array

    """
    q[q[:, 3] < 0, :] = -q[q[:, 3] < 0, :]
    return q


def dcm_to_ea313(dcm: np.ndarray) -> np.ndarray:
    """Finds the Euler angle (3-1-3) body frame sequence corresponding
    to the input direction cosine matrices

    Args:
        dcm (np.ndarray 3x3xn): DCM array

    Returns:
        np.ndarray nx3: Euler angle array

    """
    eas = np.zeros((dcm.shape[2], 3))
    eas[:, 0] = atan2d(dcm[2, 0, :], -dcm[2, 1, :])
    eas[:, 1] = acosd(dcm[2, 2, :])
    eas[:, 2] = atan2d(dcm[0, 2, :], dcm[1, 2, :])
    return eas


def ea_to_dcm(seq: Tuple, a1: np.ndarray, a2: np.ndarray, a3: np.ndarray) -> np.ndarray:
    rs = axis_rotation_matrices()
    r3 = rs[seq[0] - 1](a3)
    r2 = rs[seq[1] - 1](a2)
    r1 = rs[seq[2] - 1](a1)
    dcms = np.moveaxis(
        np.moveaxis(r1, -1, 0) @ np.moveaxis(r2, -1, 0) @ np.moveaxis(r3, -1, 0), 0, -1
    )
    return dcms


def dcm_to_quat(c: np.ndarray) -> np.ndarray:
    tr = np.trace(c, axis1=0, axis2=1)
    if c.ndim == 2:
        n = 1
        c = np.reshape(c, (3, 3, 1))
    elif c.ndim == 3:
        n = c.shape[2]

    e1_sq = 1 / 4 * (1 + 2 * c[0, 0, :] - tr)  # finds e1^2
    e2_sq = 1 / 4 * (1 + 2 * c[1, 1, :] - tr)  # finds e2^2
    e3_sq = 1 / 4 * (1 + 2 * c[2, 2, :] - tr)  # finds e3^2
    e4_sq = 1 / 4 * (1 + tr.flatten())  # finds e4^2

    q = np.zeros((c.shape[2], 4))
    test_vals = np.reshape((e1_sq, e2_sq, e3_sq, e4_sq), (4, n)).T
    # combines all values together

    best = np.argmin(test_vals, axis=1)
    tol = 1e-10
    (b1, b2, b3, b4) = (best == 0, best == 1, best == 2, best == 3)

    q[b1, 0] = np.sqrt(e1_sq[b1])
    q[b1, 1] = (c[0, 1, b1] + c[1, 0, b1]) / (4 * q[b1, 0])
    q[b1, 2] = (c[2, 0, b1] + c[0, 2, b1]) / (4 * q[b1, 0])
    q[b1, 3] = (c[1, 2, b1] - c[2, 1, b1]) / (4 * q[b1, 0])

    q[b2, 1] = np.sqrt(e2_sq[b2])
    q[b2, 0] = (c[0, 1, b2] + c[1, 0, b2]) / (4 * q[b2, 1])
    q[b2, 2] = (c[1, 2, b2] + c[2, 1, b2]) / (4 * q[b2, 1])
    q[b2, 3] = (c[2, 0, b2] - c[0, 2, b2]) / (4 * q[b2, 1])

    q[b3, 2] = np.sqrt(e3_sq[b3])
    q[b3, 0] = (c[2, 0, b3] + c[0, 2, b3]) / (4 * q[b3, 2])
    q[b3, 1] = (c[1, 2, b3] + c[2, 1, b3]) / (4 * q[b3, 2])
    q[b3, 3] = (c[0, 1, b3] - c[1, 0, b3]) / (4 * q[b3, 2])

    q[b4, 3] = np.sqrt(e4_sq[b4])
    q[b4, 0] = (c[1, 2, b4] - c[2, 1, b4]) / (4 * q[b4, 3])
    q[b4, 1] = (c[2, 0, b4] - c[0, 2, b4]) / (4 * q[b4, 3])
    q[b4, 2] = (c[0, 1, b4] - c[1, 0, b4]) / (4 * q[b4, 3])

    return q


def analytic_torque_free_attitude(
    quat0: np.ndarray,
    omega: np.ndarray,
    itensor: np.ndarray,
    teval: np.ndarray,
    ksquared: np.ndarray,
    tau0: float,
    tau_dot: float,
    is_sam: bool,
    itensor_org_factor: int = 1,
    itensor_inv_inds: list = [0, 1, 2],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Analytically propagates the orientation of an object
    under torque-free rotation forward in time.

      Args:
        quat0 - 4x1 array, unit quaternion rotating from the body frame
            to the inertial frame at the initial time step (4th
            entry is the scalar component of the quaternion)
        omega - 3xN array, body-frame angular velocity vectors at each
            point in teval. Each column is a vector, unit rad/s
        itensor - 3x3 array, Pitensor inertia tensor arranged such that
            itensor[0,0] <= itensor[1,1] <= itensor[2,2]
        teval - Nx1 array, time steps to output at. teval(1)
            corresponds to q_init and omega0
        is_sam - scalar, indicates the rotation mode. 0 = long-axis
            mode (LAM), 1 = short-axis mode (SAM)

      Output:
        q_list - 4xN array, unit quaternions rotating from the body
            frame to the inertial frame at each point in teval.
            Each column is a unit quaternion

    Author: Alexander Burton in MATLAB
    Created: January 12, 2023
    Adapted for python by Liam Robinson, March 3, 2023

    """
    # Compute angular momentum vector and direction
    hvec = (itensor @ omega.T).T
    hmag = np.linalg.norm(hvec[0, :])
    hhat = hat(hvec[0, :])

    # separate itensor components
    (ix, iy, iz) = np.diag(itensor)

    # separate out the angular velocity components
    (wx, wy, wz) = (omega[:, 0], omega[:, 1], omega[:, 2])

    # get initial nutation angle
    if is_sam:
        theta0 = np.arccos(hhat[2])
    else:
        theta0 = np.arccos(hhat[0])

    (cT, sT, cT2, sT2) = (
        np.cos(theta0),
        np.sin(theta0),
        np.cos(theta0 / 2),
        np.sin(theta0 / 2),
    )

    # get initial spin angle
    if is_sam:
        psi0 = np.arctan2(hhat[0] / sT, hhat[1] / sT)
    else:
        psi0 = np.arctan2(hhat[2] / sT, -hhat[1] / sT)

    (cS, sS) = (np.cos(psi0), np.sin(psi0))
    (cS2, sS2) = (np.cos(psi0 / 2), np.sin(psi0 / 2))

    # get initial precession angle
    A = (cT + 1) * cT2 * cS2 + sT * sT2 * (cS * cS2 + sS * sS2)
    B = -(cT + 1) * cT2 * sS2 + sT * sT2 * (cS * sS2 - sS * cS2)
    C = np.sqrt(2 * (cT + 1))

    ti = np.real((B - np.sqrt(A**2 + B**2 - C**2 + 0j)) / (A + C))
    phi0 = 4 * np.arctan(ti)

    # compute the nutation and spin angles at each time step
    if is_sam:
        theta = np.arccos(iz * wz / hmag)
        psi = np.arctan2(ix * wx, iy * wy)
    else:
        theta = np.arccos(ix * wx / hmag)
        psi = np.arctan2(iz * wz, -iy * wy)

    # get Lambda value used to compute phi
    if is_sam:
        phi_denom = -iz * (ix - iy) / (ix * (iy - iz))
    else:
        phi_denom = -ix * (iz - iy) / (iz * (iy - ix))

    (_, _, _, amp0) = scipy.special.ellipj(tau0, ksquared)
    phi_lambda = -elliptic_pi_incomplete(phi_denom, amp0, ksquared)

    # compute the precession angles
    t = itensor_org_factor * (teval - teval[0])
    (_, _, _, ampt) = scipy.special.ellipj(tau_dot * t + tau0, ksquared)
    phi_pi = elliptic_pi_incomplete(phi_denom, ampt, ksquared)

    if is_sam:
        phi = (
            phi0
            + hmag / iz * t
            + hmag * (iz - ix) / (tau_dot * ix * iz) * (phi_pi + phi_lambda)
        )
    else:
        phi = (
            hmag / ix * t
            + phi0
            + hmag * (ix - iz) / (tau_dot * ix * iz) * (phi_pi + phi_lambda)
        )

    quat_r = quat_r_from_eas(hhat, phi, theta, psi, is_sam).T
    quat_r[:, [0, 1, 2]] = quat_r[:, itensor_inv_inds]
    if itensor_org_factor == 1:
        quat_r = quat_inv(quat_r)
    # Compute the quaternions corresponding to the Euler angles
    quat_bi = quat_add(quat0, quat_r)
    return (phi, theta, psi, quat_bi)


def quat_r_from_eas(hhat, phi, theta, psi, is_sam):
    (hx, hy, hz) = hhat
    (st2, ct2) = (np.sin(theta / 2), np.cos(theta / 2))
    (sm, sp, cm, cp) = (
        np.sin((phi - psi) / 2),
        np.sin((phi + psi) / 2),
        np.cos((phi - psi) / 2),
        np.cos((phi + psi) / 2),
    )

    if is_sam:
        return (
            1
            / np.sqrt(2 * (hz + 1))
            * np.array(
                [
                    -(hz + 1) * st2 * cm - ct2 * (hx * sp - hy * cp),
                    -(hz + 1) * st2 * sm - ct2 * (hx * cp + hy * sp),
                    -(hz + 1) * ct2 * sp + st2 * (hx * cm + hy * sm),
                    (hz + 1) * ct2 * cp + st2 * (hy * cm - hx * sm),
                ]
            )
        )
    else:
        return (
            1
            / np.sqrt(2 * (hx + 1))
            * np.array(
                [
                    -(hx + 1) * ct2 * sp + st2 * (hz * cm - hy * sm),
                    (hx + 1) * st2 * sm + ct2 * (hz * cp - hy * sp),
                    -(hx + 1) * st2 * cm - ct2 * (hy * cp + hz * sp),
                    (hx + 1) * ct2 * cp - st2 * (hy * cm + hz * sm),
                ]
            )
        )


def analytic_torque_free_angular_velocity(
    omega0: np.ndarray,
    itensor: np.ndarray,
    teval: np.ndarray,
    itensor_org_factor: int = 1,
):
    """Analytically propogates the angular velocity of an object
        assuming no torque is being applied.

       Args:
          omega0 - 3x1 array, initial angular velocity in the body-fixed frame
          itensor - 3x3 array, body's principal moments of inertia (iz < iy < ix)
          teval - nx1 array, times to compute angular velocity at

       Returns:
          w_out - 3xn array, angular velocity components at each time relative
             to the body-fixed frame
          rot_mode - string ("Long Axis", "Short Axis", "Edge"), the mode of
             rotation the body is undergoing
          ksq - scalar, k-squared value used to compute angular velocity
          coef_list - 3x1 array, list of coefficients used to compute angular
             velocity
          tau0 - scalar, value of parameter tau at time teval(1)
          tau_dot - scalar, rate of change of tau with respect to time

    Author: Alexander Burton
    Created: March 22, 2022
    Edited: April 1, 2022 - converted from I3 < I1 < I2 to iz < iy < ix
            December 8, 2022 - change to ix < iy < iz, add tau_dot

    Adapted for Python by Liam Robinson, March 3, 2023

    """
    (wx0, wy0, wz0) = omega0  # Break up initial angular velocity
    # Angular momentum
    hvec = itensor @ omega0
    hmag = np.linalg.norm(hvec)

    # Rotational kinetic energy
    T = 0.5 * omega0.T @ itensor @ omega0

    idyn = hmag**2 / (2 * T)  # Dynamic moment of inertia
    we = 2 * T / hmag  # Effective angular velocity

    (ix, iy, iz) = np.diag(itensor)

    if idyn == iy:
        ValueError("Edge Case! Needs to be implemented")
    elif idyn < iy:  # For long axis mode rotation
        is_sam = False
        (ix, iz) = (iz, ix)
        (wx0, wz0) = (wz0, wx0)
    else:
        is_sam = True

    ibig_neg = 2 * int(wz0 > 0) - 1  # 1 if not negative, -1 else

    tau_dot = we * np.sqrt(idyn * (idyn - ix) * (iz - iy) / (ix * iy * iz))
    ksquared = (iy - ix) * (iz - idyn) / ((iz - iy) * (idyn - ix))
    cos_phi = np.sqrt(ix * (iz - ix) / (idyn * (iz - idyn))) * wx0 / we
    sin_phi = np.sqrt(iy * (iz - iy) / (idyn * (iz - idyn))) * wy0 / we

    psi = ibig_neg * np.arctan2(sin_phi, cos_phi)
    tau0 = scipy.special.ellipkinc(psi, ksquared)
    tau = itensor_org_factor * tau_dot * (teval - teval[0]) + tau0
    (sn, cn, dn, _) = scipy.special.ellipj(tau, ksquared)

    wx = we * np.sqrt(idyn * (iz - idyn) / (ix * (iz - ix))) * cn
    wy = ibig_neg * we * np.sqrt(idyn * (iz - idyn) / (iy * (iz - iy))) * sn
    wz = ibig_neg * we * np.sqrt(idyn * (idyn - ix) / (iz * (iz - ix))) * dn

    if not is_sam:
        (wz, wx) = (wx, wz)

    omega = np.reshape((wx, wy, wz), (3, wx.size)).T
    return (omega, ksquared, tau0, tau_dot, is_sam)


def propagate_attitude_torque_free(
    quat0: np.ndarray, omega0: np.ndarray, itensor: np.ndarray, teval: np.ndarray
) -> np.ndarray:
    """Computes torque free motion for a arbitrary inertia tensor
    using elliptic integrals and Jacobi elliptic functions

    Args:
        quat0 (np.ndarray 1x4): Initial orientation
        omega0 (np.ndarray 1x3): Initial angular velocity [rad/s]
        itensor (np.ndarray 3x3): Inertia tensor in principal axes [kg m^2]
        teval (np.ndarray nx1): Times to evaluate at past initial state [s]

    Returns:
        np.ndarray nx4: Orientation of body over time as a quaternion
        np.ndarray nx3: Angular velocity of body over time [rad/s]
    """
    is_spherical_itensor = np.unique(np.diag(itensor)).size == 1
    is_single_axis = (
        vecnorm(hat(omega0) - np.array([[1, 0, 0]])) < 1e-14 or
        vecnorm(hat(omega0) - np.array([[0, 1, 0]])) < 1e-14 or
        vecnorm(hat(omega0) - np.array([[0, 0, 1]])) < 1e-14
    )

    # Checking for single-axis rotation cases
    if is_spherical_itensor or is_single_axis:
        # If spherically symmetric
        t = teval - teval[0]
        omega = np.tile(omega0, (teval.size, 1))
        quat_from_initial = rv_to_quat(np.expand_dims(t, 1) * omega)
        quat = quat_add(np.tile(quat0, (teval.size, 1)), quat_from_initial)
        return (quat, omega)
    
    itensor_inds = np.argsort(np.diag(itensor))
    itensor_inv_inds = np.argsort(itensor_inds)
    moved_inds = np.argwhere(itensor_inds != [0, 1, 2]).flatten()
    itensor_org_factor = 2 * (moved_inds.size % 2 or moved_inds.size == 0) - 1
    # Figures out how itensor should be organized to satisfy Ix <= Iy <= Iz
    itensor = np.diag(np.diag(itensor)[itensor_inds])
    omega0 = omega0[itensor_inds]

    (omega, ksquared, tau0, tau_dot, is_sam) = analytic_torque_free_angular_velocity(
        omega0, itensor, teval, itensor_org_factor
    )
    (phi, theta, psi, quat) = analytic_torque_free_attitude(
        quat0=quat0,
        omega=omega,
        itensor=itensor,
        teval=teval,
        ksquared=ksquared,
        tau0=tau0,
        tau_dot=tau_dot,
        is_sam=is_sam,
        itensor_inv_inds=itensor_inv_inds,
        itensor_org_factor=itensor_org_factor,
    )

    omega = omega[:, itensor_inv_inds]

    return (quat, omega)
