
import numpy as np
from .astro_const import AstroConstants
from .math_utility import *
from .attitude_lib import axis_rotation_matrices
import scipy

def two_body_dynamics(rv: np.ndarray) -> np.ndarray:
    rvdot = np.empty(rv.shape)
    rvdot[:3] = rv[3:]
    rvdot[3:] = -AstroConstants.earth_mu * rv[:3] \
        / np.linalg.norm(rv[:3])**3
    return rvdot

def j2_acceleration(rvec: np.ndarray) -> np.ndarray:
    re = AstroConstants.earth_r_eq
    mu = AstroConstants.earth_mu
    j2 = AstroConstants.earth_j2
    r = np.linalg.norm(rvec)
    (x_eci, y_eci, z_eci) = rvec # ECI positions
    return -3/2 * j2 * (mu/r**2) * (re/r)**2 \
        * np.array([(1 - 5 * (z_eci/r)**2) * x_eci/r,
                    (1 - 5 * (z_eci/r)**2) * y_eci/r,
                    (3 - 5 * (z_eci/r)**2) * z_eci/r])

def rv_to_coe(a: float, e: float, i: float, Om: float, om: float, 
              ma: float, mu: float = AstroConstants.earth_mu) -> np.array:
    efun = lambda ea: np.deg2rad(ma) - np.deg2rad(ea) + e * sind(ea)
    ea = scipy.optimize.fsolve(efun, ma) # Eccentric anomaly
    ta = wrap_to_360(2 * atand(tand(ea/2) * np.sqrt((1+e)/(1-e)))) # True anomaly
    ta = float(ta) # TODO: accept vector arguments 
    p = a * (1-e**2) # Semi-latus rectum
    h = np.sqrt(p*mu) # Angular momentum
    r = p / (1+e*cosd(ta)) # Position magnitude
    v = np.sqrt(mu * (2/r-1/a)) # Velocity magnitude
    isasc = 2 * (ta < 180) - 1 # 1 if asc, -1 if not
    y = isasc * acosd(h / (r*v)) # Flight path angle above/below local horizon
    r_rth = r * np.array([1, 0, 0]) # Rotating frame position
    v_rth = v * np.array([sind(y), cosd(y), 0]) # Rotating frame velocity
    
    (R1, _, R3) = axis_rotation_matrices() # Getting basic DCMs
    D_i2r = R3(np.deg2rad(ta+om)) @ R1(np.deg2rad(i)) @ R3(np.deg2rad(Om))
    # DCM from inertial to rotating frame
    r_i = np.transpose(D_i2r) @ r_rth # Rotating rv to inertial space
    v_i = np.transpose(D_i2r) @ v_rth
    return (r_i, v_i)
