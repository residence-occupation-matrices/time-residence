import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from time_residence.model_inputs import ModelInput, build_model_input


class ModelInputTests(unittest.TestCase):
    def test_alpha_and_rom_follow_the_published_conditioning(self):
        residences = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "residence_ageb_index": [0, 0, 1],
            }
        )
        pings = pd.DataFrame(
            {
                "id": ["a", "a", "b", "b", "c", "c"],
                "ageb_index": [0, 1, 0, 0, 1, 1],
            }
        )
        metadata = pd.DataFrame(
            {
                "ageb_index": [0, 1],
                "CVE_AGEB": ["0001", "0002"],
                "POBTOT": [1000, 2000],
            }
        )
        individual = {
            "a": np.array([0.2, 0.8]),
            "b": np.array([1.0, 0.0]),
            "c": np.array([0.0, 1.0]),
        }
        result = build_model_input("P1A", individual, pings, residences, metadata)
        np.testing.assert_allclose(result.alpha, [0.5, 0.0])
        np.testing.assert_allclose(result.rom, [[0.2, 0.8], [0.0, 1.0]])

    def test_npz_round_trip_preserves_alignment(self):
        item = ModelInput(
            period="P1A",
            ageb_ids=np.array(["0001", "0002"]),
            rom=np.array([[0.8, 0.2], [0.1, 0.9]]),
            alpha=np.array([0.4, 0.6]),
            population=np.array([1000.0, 2000.0]),
            metadata={"note": "test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "P1A.npz"
            item.save(path)
            loaded = ModelInput.load(path)
        self.assertEqual(loaded.period, item.period)
        self.assertEqual(loaded.metadata, item.metadata)
        np.testing.assert_array_equal(loaded.ageb_ids, item.ageb_ids)
        np.testing.assert_allclose(loaded.rom, item.rom)
        np.testing.assert_allclose(loaded.alpha, item.alpha)
        np.testing.assert_allclose(loaded.population, item.population)


if __name__ == "__main__":
    unittest.main()
