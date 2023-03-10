from datetime import timezone, datetime, timedelta
import numpy as np
from typing import Any


def beginning_of_day(dates: np.ndarray[datetime]) -> np.ndarray[datetime]:
    """Finds

    Args:
        dates (np.ndarray[datetime] nx1) [UTC]: Date array

    Returns:
        np.ndarray[datetime] nx1 [UTC]: Beginning of day for each date

    """
    return_naked = False
    if isinstance(dates, datetime):
        dates = np.array([dates])
        return_naked = True

    bod_arr = np.array(
        [datetime(d.year, d.month, d.day, 0, 0, 0, 0, tzinfo=d.tzinfo) for d in dates]
    )

    if return_naked:
        return bod_arr[0]
    else:
        return bod_arr


def date_to_ut(date) -> float:
    """Converts a datetime object to Universal Time (UT)

    Args:
        date (datetime) [UTC]: Date object to compute UT for

    Returns:
        float [hr]: UT at input date

    """

    min_year = np.min([d.year for d in np.array(date).flatten()])
    assert min_year > 1582, ValueError("date must be after 1582")
    bod = beginning_of_day(date)
    # Generates the datetime object at the beginning of the day
    univ_delta = date - bod
    # Calculates the decimal hours since the beginning of the day
    if hasattr(univ_delta, "__iter__"):
        return np.array([dt.total_seconds() / 3600 for dt in univ_delta])
    else:
        return univ_delta.total_seconds() / 3600


def date_to_jd(date: Any) -> float:
    """Converts datetime to Julian date

    Args:
        date (datetime) [UTC]: Date object to compute UT for

    Returns:
        float: Julian date of input date

    """
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
    """Converts a datetime to sidereal time

    Args:
        date (datetime) [UTC]: Date object to compute UT for

    Returns:
        float: Sidereal time at the input date

    """
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


def jd_now() -> float:
    """Computes the Julian date at function runtime"""
    return date_to_jd(now())


def now() -> datetime:
    """Returns the current date object at runtime, set to UTC"""
    return datetime.now(tz=timezone.utc)


def date_linspace(
    date_start: datetime, date_stop: datetime, num: int
) -> np.ndarray[datetime]:
    """Computes a linspace of datetime objects

    Args:
        date_start (datetime.datetime) [UTC]: Date object to start at
        date_stop (datetime.datetime) [UTC]: Date object to stop at
        num (int): Number of samples to make

    Returns:
        np.ndarray[datetime]: Sampled linspace of datetimes

    """
    delta_seconds = (date_stop - date_start).total_seconds() / (int(num) - 1)
    return np.array(
        [date_start + timedelta(seconds=n * delta_seconds) for n in range(int(num))]
    )
