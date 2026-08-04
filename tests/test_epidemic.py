import tempfile
import unittest
from pathlib import Path

import numpy as np

from time_residence.epidemic import (
    EpidemicParameters,
    compare_period,
    effective_rom,
    initial_state,
    reproduce_figure8,
    seirs_rhs,
    write_synthetic_figure8_inputs,
)
from time_residence.model_inputs import synthetic_model_input


class EpidemicTests(unittest.TestCase):
    def test_effective_rom_is_row_stochastic(self):
        matrix = np.array([[0.75, 0.25], [0.4, 0.6]])
        alpha = np.array([0.2, 0.8])
        result = effective_rom(matrix, alpha)
        expected = np.diag(alpha) @ matrix + np.diag(1.0 - alpha)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_initial_state_requires_every_requested_seed(self):
        model_input = synthetic_model_input("P1A", n_patches=3, seed=3)
        with self.assertRaisesRegex(ValueError, "do not occur"):
            initial_state(model_input, seed_agebs=["A0000", "missing"])

    def test_rhs_matches_the_original_dense_matrix_expression(self):
        matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
        alpha = np.array([0.4, 0.6])
        pstar = effective_rom(matrix, alpha)
        population = np.array([1000.0, 1500.0])
        parameters = EpidemicParameters()
        susceptible = np.array([990.0, 1490.0])
        exposed = np.array([4.0, 5.0])
        infected = np.array([6.0, 5.0])
        recovered = np.zeros(2)
        state = np.concatenate([susceptible, exposed, infected, recovered])

        actual = seirs_rhs(state, 0.0, pstar, population, parameters).reshape(4, 2)
        present_population = pstar.T @ population
        dense_new_exposures = (
            np.diag(susceptible)
            @ pstar
            @ np.diag(np.full(2, parameters.beta))
            @ np.linalg.inv(np.diag(present_population))
            @ pstar.T
            @ infected
        )
        expected_s = (
            parameters.mu * population
            - dense_new_exposures
            - parameters.mu * susceptible
            + parameters.tau * recovered
        )
        expected_e = dense_new_exposures - (parameters.kappa + parameters.mu) * exposed
        expected_i = (
            parameters.kappa * exposed
            - (parameters.gamma + parameters.psi + parameters.mu) * infected
        )
        expected_r = parameters.gamma * infected - (parameters.tau + parameters.mu) * recovered
        np.testing.assert_allclose(actual, [expected_s, expected_e, expected_i, expected_r])

    def test_period_comparison_evaluates_near_day_30(self):
        first = synthetic_model_input("P1A", n_patches=5, seed=1)
        second = synthetic_model_input("P1B", n_patches=5, seed=2)
        identifiers = np.array([f"A{i:04d}" for i in range(5)])
        first.ageb_ids = identifiers
        second.ageb_ids = identifiers
        time = np.linspace(0.0, 200.0, 100)
        result = compare_period("P1", first, second, time, seed_agebs=["A0000"])
        self.assertAlmostEqual(result.evaluation_day, time[np.argmin(abs(time - 30.0))])
        self.assertEqual(result.per_ageb_difference.shape, (100, 5))

    def test_synthetic_end_to_end_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            seeds = write_synthetic_figure8_inputs(input_dir, n_patches=4, seed=10)
            results = reproduce_figure8(
                input_dir,
                output_dir,
                seed_agebs=seeds,
                time_end=40.0,
                time_points=21,
            )
            self.assertEqual(len(results), 3)
            self.assertTrue((output_dir / "figure8.png").is_file())
            self.assertTrue((output_dir / "figure8_summary.json").is_file())
            for label in ("P1", "P2", "P3"):
                self.assertTrue((output_dir / f"figure8_{label}.npz").is_file())


if __name__ == "__main__":
    unittest.main()
