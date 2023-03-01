import numpy as np
from .astro_const import AstroConstants
from .math_utility import sph_to_cart
from .astro_coordinates import ecef_to_eci


def geodetic_lat_to_geocentric(lat_geod_rad: np.ndarray) -> np.ndarray:
    f = AstroConstants.earth_f
    return np.arctan((1 - f) ** 2 * np.tan(lat_geod_rad))


def radius_at_geodetic_lat(lat_geodetic_rad: float) -> float:
    lat_geoc = geodetic_lat_to_geocentric(lat_geodetic_rad)
    # Transforms geodetic latitude into geocentric
    return AstroConstants.earth_r_eq - 21.38 * np.sin(lat_geoc) ** 2


def lla_to_itrf(lat_geod_rad: float, lon_rad: float, a_km: float) -> np.array:
    lat_geoc = geodetic_lat_to_geocentric(lat_geod_rad)
    # Transforms geodetic latitude into geocentric

    r_earth_at_lat = radius_at_geodetic_lat(lat_geod_rad)
    # Computes the radius of the earth at the given geodetic latitude
    r_topo = r_earth_at_lat + a_km
    # Computes the altitude of the observer at this point [km]

    (x_itrf, y_itrf, z_itrf) = sph_to_cart(lon_rad, lat_geoc, r_topo)
    return np.array((x_itrf, y_itrf, z_itrf))


def lla_to_eci(lat_geod_rad: float, lon_rad: float, a_km: float, date) -> np.array:
    r_itrf = lla_to_itrf(lat_geod_rad, lon_rad, a_km)
    sidereal_rot = ecef_to_eci(date)
    return sidereal_rot @ r_itrf
