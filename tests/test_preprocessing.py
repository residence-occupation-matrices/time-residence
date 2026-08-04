import unittest

import pandas as pd

from time_residence.preprocessing import (
    classify_leavers,
    infer_residences,
    normalize_ping_columns,
    prepare_agebs,
)


class SpatialFrame(pd.DataFrame):
    _metadata = ["crs"]

    @property
    def _constructor(self):
        return SpatialFrame


class Geometry:
    is_empty = False


class PreprocessingTests(unittest.TestCase):
    def test_ageb_index_is_stable_and_code_sorted(self):
        agebs = SpatialFrame(
            {
                "CVE_AGEB": ["B", "A"],
                "geometry": [Geometry(), Geometry()],
            }
        )
        agebs.crs = "EPSG:4326"
        result = prepare_agebs(agebs)
        self.assertEqual(result["CVE_AGEB"].tolist(), ["A", "B"])
        self.assertEqual(result["ageb_index"].tolist(), [0, 1])

    def test_reversed_coordinate_headers_require_the_repair_option(self):
        frame = pd.DataFrame(
            {
                "id_adv": ["x"],
                "timestamp": ["2020-09-21T12:00:00Z"],
                "lat": [-110.95],
                "lon": [29.08],
            }
        )
        with self.assertRaisesRegex(ValueError, "header inversion"):
            normalize_ping_columns(frame)
        repaired = normalize_ping_columns(frame, repair_swapped_coordinates=True)
        self.assertAlmostEqual(repaired.loc[0, "latitude"], 29.08)
        self.assertAlmostEqual(repaired.loc[0, "longitude"], -110.95)

    def test_residence_tie_breaking_is_deterministic(self):
        pings = pd.DataFrame(
            {
                "id": ["device"] * 4,
                "timestamp": pd.to_datetime(
                    [
                        "2020-09-22T05:00:00Z",
                        "2020-09-22T06:00:00Z",
                        "2020-09-22T20:00:00Z",
                        "2020-09-22T21:00:00Z",
                    ],
                    utc=True,
                ),
                "latitude": [29.0] * 4,
                "longitude": [-111.0] * 4,
                "ageb_index": [0, 1, 0, 1],
            }
        )
        metadata = pd.DataFrame(
            {
                "ageb_index": [0, 1],
                "CVE_AGEB": ["0001", "0002"],
                "POBTOT": [100, 900],
            }
        )
        first = infer_residences(pings, metadata, seed=7)
        second = infer_residences(pings.sample(frac=1.0), metadata, seed=7)
        self.assertEqual(
            first.loc[0, "residence_ageb_index"],
            second.loc[0, "residence_ageb_index"],
        )

    def test_leaver_definition_uses_any_observed_ping_outside_home(self):
        pings = pd.DataFrame(
            {
                "id": ["a", "a", "b", "b"],
                "ageb_index": [0, 1, 1, 1],
            }
        )
        residences = pd.DataFrame({"id": ["a", "b"], "residence_ageb_index": [0, 1]})
        result = classify_leavers(pings, residences).set_index("id")
        self.assertTrue(result.loc["a", "leaves_home"])
        self.assertFalse(result.loc["b", "leaves_home"])


if __name__ == "__main__":
    unittest.main()
