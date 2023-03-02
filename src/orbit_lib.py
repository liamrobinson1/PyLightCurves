import numpy as np
from .astro_const import AstroConstants
from .math_utility import *
from .attitude_lib import axis_rotation_matrices
import scipy
from typing import Callable
from .attitude_lib import quat_kinematics, rigid_rotation_dynamics, quat_to_dcm
from .astro_coordinates import sun


def integrate_orbit_and_attitude(
    rv0: np.ndarray,
    q0: np.ndarray,
    w0: np.ndarray,
    itensor: np.ndarray,
    orbit_perturbations: Callable,
    body_torque: Callable,
    teval: np.ndarray,
    int_tol: float = 1e-13,
) -> np.ndarray:
    fun = lambda t, y: np.concatenate(
        (
            two_body_dynamics(y[0:6])
            + np.concatenate(([0, 0, 0], orbit_perturbations(t, y[0:6]))),
            np.concatenate(
                (
                    quat_kinematics(y[6:10], y[10:13]),
                    rigid_rotation_dynamics(
                        t,
                        y[10:13],
                        itensor,
                        lambda t: body_torque(t, quat_to_dcm(y[6:10]).T @ y[0:3]),
                    ),
                )
            ),
        )
    )
    y0 = np.concatenate((rv0, q0, w0))
    tspan = [np.min(teval), np.max(teval)]
    ode_res = scipy.integrate.solve_ivp(
        fun, tspan, y0, t_eval=teval, rtol=int_tol, atol=int_tol
    )
    ode_res = ode_res.y.T
    # Returns (r, v, q, w)
    return (ode_res[:, 0:3], ode_res[:, 3:6], ode_res[:, 6:10], ode_res[:, 10:13])


def integrate_orbit_dynamics(
    rv0: np.ndarray, perturbations: Callable, teval: np.ndarray, int_tol=1e-13
) -> np.ndarray:
    """Integration for orbital dynamics with perturbations

    Args:
        rv0 (np.ndarray 1x6): Initial position and velocity state in Earth-centered inertial coordinates
        pertubations (Callable(t,rv0) -> np.ndarray 1x3) [Nm]: Accelerations applied to the body due to perturbing forces
        teval (np.ndarray 1xn) [seconds]: Times to return integrated trajectory at
        int_tol (float): Integration rtol and atols for RK45

    Returns:
        np.ndarray nx7: Integrated quaternion [:4, :] and angular velocity [4:, :]

    """
    fun = lambda t, y: two_body_dynamics(y) + np.concatenate(
        ([0, 0, 0], perturbations(t, y))
    )
    tspan = [np.min(teval), np.max(teval)]
    ode_res = scipy.integrate.solve_ivp(
        fun, tspan, rv0, t_eval=teval, rtol=int_tol, atol=int_tol
    )
    return ode_res.y.T


def sun_acceleration(rv: np.ndarray, jd: float) -> np.ndarray:
    """Approximate 3rd body acceleration due to the Sun

    Args:
        rv (np.ndarray 1x6): Position and velocity state in Earth-centered inertial coordinates
        jd (float) [days]: Julian date to evaluate at

    Returns:
        np.ndarray 1x6: Time derivative of position and velocity states

    """
    earth_to_sun = sun(jd).flatten() * AstroConstants.au_to_km
    sat_to_sun = earth_to_sun - rv[0:3]
    return -AstroConstants.sun_mu * (
        earth_to_sun / np.linalg.norm(earth_to_sun) ** 3
        - sat_to_sun / np.linalg.norm(sat_to_sun) ** 3
    )


def two_body_dynamics(rv: np.ndarray) -> np.ndarray:
    """Relative two-body dynamics for orbits

    Args:
        rv (np.ndarray 1x6) [km, km/s]: Position [:3] and velocity [3:] of the satellite

    Returns:
        np.ndarray 1x6: Time derivative of position and velocity states

    """
    rvdot = np.empty(rv.shape)
    rvdot[:3] = rv[3:]
    rvdot[3:] = -AstroConstants.earth_mu * rv[:3] / np.linalg.norm(rv[:3]) ** 3
    return rvdot


def j2_acceleration(rvec: np.ndarray) -> np.ndarray:
    """Acceleration due to the J2 perturbation (Earth oblateness)

    Args:
        rvec (np.ndarray 1x3) [km]: Position vector of the satellite

    Returns:
        np.ndarray 1x3 [km/s^2]: Acceleration due to J2

    """
    re = AstroConstants.earth_r_eq
    mu = AstroConstants.earth_mu
    j2 = AstroConstants.earth_j2
    r = np.linalg.norm(rvec)
    (x_eci, y_eci, z_eci) = rvec  # ECI positions
    return (
        -3
        / 2
        * j2
        * (mu / r**2)
        * (re / r) ** 2
        * np.array(
            [
                (1 - 5 * (z_eci / r) ** 2) * x_eci / r,
                (1 - 5 * (z_eci / r) ** 2) * y_eci / r,
                (3 - 5 * (z_eci / r) ** 2) * z_eci / r,
            ]
        )
    )


def coe_to_rv(
    a: float,
    e: float,
    i: float,
    Om: float,
    om: float,
    ma: float,
    mu: float = AstroConstants.earth_mu,
) -> np.array:
    """Converts classical (Keplerian) orbital elements to position/velocity state

    Args:
        a (float) [km]: Semi-major axis of the orbit
        e (float): Eccentricity
        i (float) [deg]: Inclination
        Om (float) [deg]: Right Ascension of the Ascending Node (RAAN)
        om (float) [deg]: Argument of periapsis
        ma (float) [deg]: Mean anomaly
        mu (float) [km^3/s^2]: Gravitational parameter of central body

    Returns:
        np.array 1x6 [km, km/s]: State vector in position/velocity

    """
    efun = lambda ea: np.deg2rad(ma) - np.deg2rad(ea) + e * sind(ea)
    ea = scipy.optimize.fsolve(efun, ma)  # Eccentric anomaly
    ta = wrap_to_360(
        2 * atand(tand(ea / 2) * np.sqrt((1 + e) / (1 - e)))
    )  # True anomaly
    ta = float(ta)  # TODO: accept vector arguments
    p = a * (1 - e**2)  # Semi-latus rectum
    h = np.sqrt(p * mu)  # Angular momentum
    r = p / (1 + e * cosd(ta))  # Position magnitude
    v = np.sqrt(mu * (2 / r - 1 / a))  # Velocity magnitude
    isasc = 2 * (ta < 180) - 1  # 1 if asc, -1 if not
    y = isasc * acosd(h / (r * v))  # Flight path angle above/below local horizon
    r_rth = r * np.array([1, 0, 0])  # Rotating frame position
    v_rth = v * np.array([sind(y), cosd(y), 0])  # Rotating frame velocity

    (R1, _, R3) = axis_rotation_matrices()  # Getting basic DCMs
    D_i2r = R3(np.deg2rad(ta + om)) @ R1(np.deg2rad(i)) @ R3(np.deg2rad(Om))
    # DCM from inertial to rotating frame
    r_i = np.transpose(D_i2r) @ r_rth  # Rotating rv to inertial space
    v_i = np.transpose(D_i2r) @ v_rth
    return (r_i, v_i)
