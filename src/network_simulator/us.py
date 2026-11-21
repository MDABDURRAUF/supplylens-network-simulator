"""United States reference data: Census regions and the federal holiday calendar."""

from __future__ import annotations

import datetime as dt

import pandas as pd

# U.S. Census Bureau regions (50 states and the District of Columbia).
CENSUS_REGIONS = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
STATE_REGION = {state: region for region, states in CENSUS_REGIONS.items() for state in states}


def region_of(state: str) -> str:
    """Census region of a two-letter state code."""
    try:
        return STATE_REGION[str(state).strip().upper()]
    except KeyError:
        raise ValueError(f"'{state}' is not a US state code") from None


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th given weekday of a month (Monday = 0); n = -1 means the last one."""
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def federal_holidays(years) -> dict[dt.date, str]:
    """US federal holidays for the given year or years, on their actual dates.

    Returns ``{date: name}``. Use ``observed_date`` for the weekday on which a
    holiday falling at the weekend is observed.
    """
    years = [years] if isinstance(years, int) else list(years)
    out = {}
    for y in years:
        out.update({
            dt.date(y, 1, 1): "New Year's Day",
            _nth_weekday(y, 1, 0, 3): "Birthday of Martin Luther King, Jr.",
            _nth_weekday(y, 2, 0, 3): "Washington's Birthday",
            _nth_weekday(y, 5, 0, -1): "Memorial Day",
            dt.date(y, 6, 19): "Juneteenth National Independence Day",
            dt.date(y, 7, 4): "Independence Day",
            _nth_weekday(y, 9, 0, 1): "Labor Day",
            _nth_weekday(y, 10, 0, 2): "Columbus Day",
            dt.date(y, 11, 11): "Veterans Day",
            _nth_weekday(y, 11, 3, 4): "Thanksgiving Day",
            dt.date(y, 12, 25): "Christmas Day",
        })
    return out


def observed_date(day: dt.date) -> dt.date:
    """The weekday a holiday is observed: Friday if it falls on Saturday, Monday if on Sunday."""
    if day.weekday() == 5:
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


def closed_for_holiday(date, holidays: dict[dt.date, str]) -> bool:
    """Whether a site that closes for federal holidays is closed on ``date``.

    Closed on the holiday itself and, when the holiday falls at the weekend,
    on the weekday it is observed.
    """
    day = pd.Timestamp(date).date()
    return day in holidays or any(observed_date(h) == day for h in holidays)


def holiday_table(years) -> pd.DataFrame:
    """Federal holidays as a table: name, date, weekday and observed date."""
    rows = [{"holiday": name, "date": pd.Timestamp(day), "weekday": day.strftime("%A"), "observed": pd.Timestamp(observed_date(day))}
            for day, name in sorted(federal_holidays(years).items())]
    return pd.DataFrame(rows)
