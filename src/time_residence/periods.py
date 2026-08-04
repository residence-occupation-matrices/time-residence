"""Study periods and local-time filtering rules.

The source timestamps are UTC. Dates in Table 2 of the associated article are
calendar dates in Hermosillo, which remains on UTC-07:00 throughout the year.
Period filters therefore convert timestamps before applying inclusive dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import pandas as pd

HERMOSILLO_TIMEZONE: Final = "America/Hermosillo"
NIGHT_START_MINUTE: Final = 22 * 60
NIGHT_END_MINUTE: Final = 6 * 60


@dataclass(frozen=True, slots=True)
class StudyPeriod:
    """One part of a comparison period from Table 2 of the article."""

    code: str
    label: str
    start: date
    end: date

    def mask(self, timestamps: pd.Series) -> pd.Series:
        """Select timestamps whose Hermosillo calendar date is in this period.

        Both dates are inclusive. Naive timestamps are interpreted as UTC,
        matching the format of the source BigQuery records.
        """

        local = to_hermosillo_time(timestamps)
        local_date = local.dt.date
        return (local_date >= self.start) & (local_date <= self.end)


STUDY_PERIODS: Final[dict[str, StudyPeriod]] = {
    "P1A": StudyPeriod("P1A", "First period - first part", date(2020, 9, 21), date(2020, 10, 4)),
    "P1B": StudyPeriod("P1B", "First period - second part", date(2020, 10, 26), date(2020, 11, 8)),
    "P2A": StudyPeriod("P2A", "Second period - first part", date(2020, 9, 21), date(2020, 10, 4)),
    "P2B": StudyPeriod("P2B", "Second period - second part", date(2020, 11, 2), date(2020, 11, 15)),
    "P3A": StudyPeriod("P3A", "Third period - first part", date(2020, 9, 21), date(2020, 10, 11)),
    "P3B": StudyPeriod("P3B", "Third period - second part", date(2020, 10, 12), date(2020, 11, 1)),
}

PERIOD_COMPARISONS: Final[dict[str, tuple[str, str]]] = {
    "P1": ("P1A", "P1B"),
    "P2": ("P2A", "P2B"),
    "P3": ("P3A", "P3B"),
}


def to_hermosillo_time(timestamps: pd.Series) -> pd.Series:
    """Parse source timestamps as UTC and convert them to Hermosillo time."""

    parsed = pd.to_datetime(timestamps, utc=True, errors="raise")
    return parsed.dt.tz_convert(HERMOSILLO_TIMEZONE)


def night_mask(timestamps: pd.Series) -> pd.Series:
    """Select the half-open local-time interval [22:00, 06:00)."""

    local = to_hermosillo_time(timestamps)
    minute = local.dt.hour * 60 + local.dt.minute
    return (minute >= NIGHT_START_MINUTE) | (minute < NIGHT_END_MINUTE)
