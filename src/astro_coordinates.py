from .astro_time_utility import date_to_sidereal
from .attitude_lib import axis_rotation_matrices
from .math_utility import sph_to_cart
import numpy as np

def ecef_to_eci(date) -> np.ndarray:
    (_, _, r3) = axis_rotation_matrices()
    sid_time = date_to_sidereal(date)
    # Gets the current sidereal time
    gmst = (sid_time % 86400) / 86400 * 2 * np.pi
    # Gets the GMST hour angle
    return r3(-gmst) # Gets the rotation matrix about the third body axis


def sun(jd: np.array) -> np.array:
        # Translation of Vallado's MATLAB method of the same name
        if jd.size == 1: jd = np.array([jd])
        tut1 = (jd - 2451545.0) / 36525.0
        meanlong = (280.460 + 36000.77*tut1) % 360.0;
        meananomaly = np.deg2rad((357.5277233+35999.05034*tut1) % 360.0)
        meananomaly = np.array([m if m > 0 else m+2*np.pi for m in meananomaly])
        # if meananomaly < 0: meananomaly += 2*np.pi
        eclplong = np.deg2rad((meanlong + 1.914666471 * np.sin(meananomaly) \
                    + 0.019994643 * np.sin(2*meananomaly)) % 360.0)
        obliquity = np.deg2rad(23.439291 - 0.0130042*tut1)
        magr = 1.000140612 - 0.016708617 * np.cos(meananomaly) \
               - 0.000139589 *np.cos(2.0*meananomaly)
        # in AU
        rsun = np.empty((jd.shape[0], 3))
        rsun[:,0]= magr*np.cos(eclplong)
        rsun[:,1]= magr*np.cos(obliquity)*np.sin(eclplong)
        rsun[:,2]= magr*np.sin(obliquity)*np.sin(eclplong)
        return rsun
