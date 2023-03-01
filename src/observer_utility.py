from .astro_const import AstroConstants
from .math_utility import sph_to_cart
from .astro_coordinates import ecef_to_eci
import numpy as np
import datetime


def geodetic_lat_to_geocentric(lat_geod: np.ndarray) -> np.ndarray:
    """Converts geodetic latitude to geocentric latitude

    Args:
        lat_geod (np.ndarray nx1) [rad]: Geodetic latitudes

    Returns:
        np.ndarray nx1 [rad]: Geocentric latitudes

    """
    f = AstroConstants.earth_f
    return np.arctan((1 - f) ** 2 * np.tan(lat_geod))


def radius_at_geodetic_lat(lat_geodetic: np.ndarray) -> np.ndarray:
    """Earth's radius at the given geodetic latitude

    Args:
        lat_geodetic (np.ndarray nx1) [rad]: Geodetic latitudes

    Returns:
        np.ndarray nx1 [km]: Earth radius at given latitudes

    """
    lat_geoc = geodetic_lat_to_geocentric(lat_geodetic)
    return AstroConstants.earth_r_eq - 21.38 * np.sin(lat_geoc) ** 2


def lla_to_itrf(lat_geod: np.ndarray, lon: np.ndarray, a: np.ndarray) -> np.array:
    """Converts from latitude, longitude, altitude (LLA)
    to the International Terrestrial Reference Frame (ITRF)

    Args:
        lat_geod (np.ndarray nx1) [rad]: Geodetic latitudes
        lon (np.ndarray nx1) [rad]: Longitudes
        a (np.ndarray nx1) [km]: Altitudes above the WGS84 ellipsoid

    Returns:
        np.ndarray nx3 [km]: ITRF positions for each LLA triplet

    """
    lat_geoc = geodetic_lat_to_geocentric(lat_geod)
    # Transforms geodetic latitude into geocentric

    r_earth_at_lat = radius_at_geodetic_lat(lat_geod)
    # Computes the radius of the earth at the given geodetic latitude
    r_topo = r_earth_at_lat + a
    # Computes the altitude of the observer at this point [km]

    (x_itrf, y_itrf, z_itrf) = sph_to_cart(lon, lat_geoc, r_topo)
    return np.array((x_itrf, y_itrf, z_itrf))


def lla_to_eci(
    lat_geod: float, lon: float, a: float, date: datetime.datetime
) -> np.array:
    """Converts from latitude, longitude, altitude (LLA) to Earth-centered inertial (ECI) coordinates

    Args:
        lat_geod (np.ndarray nx1) [rad]: Geodetic latitudes
        lon (np.ndarray nx1) [rad]: Longitudes
        a (np.ndarray nx1) [km]: Altitudes above the WGS84 ellipsoid
        date (datetime.datetime) [UTC]: Date to evaluate conversion

    Returns:
        np.ndarray nx3 [km]: ECI positions for each LLA triplet

    """
    r_itrf = lla_to_itrf(lat_geod, lon, a)
    sidereal_rot = ecef_to_eci(date)
    return sidereal_rot @ r_itrf
