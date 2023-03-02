"""Collection of astrodynamically-relevant constants

"""


class AstroConstants:
    earth_mu = 3.9860044e5  # [km^3/s^2] Earth gravitational parameter
    sun_mu = 1.327124400e11  # [km^3/s^2] Sun gravitational parameter
    earth_r_eq = 6378.14  # [km] Earth's mean equatorial radius
    earth_j2 = 1.08262668e-3  # Coefficient of the 2,0 zonal harmonic
    earth_f = 1 / 298.257223563  # Flattening coefficient
    earth_sec_in_day = 86400  # [seconds] Seconds in a day
    au_to_km = 1.496e8  # Astronomical Units to km
