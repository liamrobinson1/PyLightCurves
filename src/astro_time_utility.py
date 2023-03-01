from datetime import timezone, datetime
import numpy as np
from typing import Any


def date_to_ut(date) -> float:
    assert date.year > 1582, ValueError("date must be after 1582")
    beginning_of_day = datetime(
        date.year, date.month, date.day, 00, 00, 00, 00, tzinfo=timezone.utc
    )
    # Generates the datetime object at the beginning of the day
    univ_delta = date - beginning_of_day
    # Calculates the decimal hours since the beginning of the day
    return univ_delta.seconds / 3600 + univ_delta.microseconds / (3600 * 1e6)


def date_to_jd(date: Any) -> float:
    ut = date_to_ut(date)
    if date.month <= 2:  # If the month is Jan or Feb
        y = date.year - 1
        m = date.month + 12
    elif date.month > 2:  # If the month is after Feb
        y = date.year
        m = date.month

    B = np.floor(y / 400) - np.floor(y / 100)  # Account for leap years
    return (
        np.floor(365.25 * y)
        + np.floor(30.6001 * (m + 1))
        + B
        + 1720996.5
        + date.day
        + ut / 24
    )


def date_to_sidereal(date) -> float:
    beginning_of_day = datetime(
        date.year, date.month, date.day, 00, 00, 00, 00, tzinfo=timezone.utc
    )
    ut = date_to_ut(date)
    jd0 = date_to_jd(beginning_of_day)
    jd = date_to_jd(date)

    T0 = (jd0 - 2451545) / 36525  # Time since Jan 1 2000, 12h UT to beginning of day
    T1 = (jd - 2451545) / 36525  # Time since Jan 1 2000, 12h UT to now

    sidereal_beginning_of_day = (
        24110.54841 + 8640184.812866 * T0 + 0.093104 * T1**2 - 0.0000062 * T1**3
    )
    # Sidereal time at the beginning of the julian date day

    # Computes the exact sidereal time, accounting for the extra 4 mins/day
    return sidereal_beginning_of_day + 1.0027279093 * ut * 3600
