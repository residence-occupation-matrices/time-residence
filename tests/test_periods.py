import unittest

import pandas as pd

from time_residence.periods import STUDY_PERIODS, night_mask


class PeriodTests(unittest.TestCase):
    def test_period_dates_are_local_and_inclusive(self):
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2020-09-21T06:59:59Z",
                    "2020-09-21T07:00:00Z",
                    "2020-10-05T06:59:59Z",
                    "2020-10-05T07:00:00Z",
                ],
                utc=True,
            )
        )
        self.assertEqual(
            STUDY_PERIODS["P1A"].mask(timestamps).tolist(),
            [False, True, True, False],
        )

    def test_night_interval_is_half_open(self):
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2020-09-22T04:59:00Z",  # 21:59 local
                    "2020-09-22T05:00:00Z",  # 22:00 local
                    "2020-09-22T12:59:00Z",  # 05:59 local
                    "2020-09-22T13:00:00Z",  # 06:00 local
                ],
                utc=True,
            )
        )
        self.assertEqual(night_mask(timestamps).tolist(), [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
