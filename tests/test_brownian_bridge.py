import unittest

import numpy as np

from time_residence.brownian_bridge import (
    ProjectedTrajectory,
    estimate_motion_parameters,
    estimate_sigma,
    gaussian_log_density,
    increment_covariance,
    negative_log_likelihood,
)


class BrownianBridgeTests(unittest.TestCase):
    def test_increment_covariance_matches_banded_formula(self):
        covariance = increment_covariance(np.array([0.0, 2.0, 5.0]), sigma=3.0, location_error=2.0)
        expected = np.array([[26.0, -4.0], [-4.0, 35.0]])
        np.testing.assert_allclose(covariance, expected)

    def test_sigma_estimator_recovers_the_standard_deviation_rate(self):
        rng = np.random.default_rng(101)
        true_sigma = 4.0
        count = 600
        increments_x = rng.normal(0.0, true_sigma, count - 1)
        increments_y = rng.normal(0.0, true_sigma, count - 1)
        trajectory = ProjectedTrajectory(
            identifier="synthetic",
            elapsed_seconds=np.arange(count, dtype=float),
            x=np.concatenate([[0.0], np.cumsum(increments_x)]),
            y=np.concatenate([[0.0], np.cumsum(increments_y)]),
        )
        estimate = estimate_sigma(trajectory, location_error=0.0)
        self.assertAlmostEqual(estimate, true_sigma, delta=0.35)

    def test_banded_likelihood_matches_dense_covariance(self):
        trajectory = ProjectedTrajectory(
            identifier="small",
            elapsed_seconds=np.array([0.0, 2.0, 5.0, 9.0]),
            x=np.array([0.0, 1.0, 4.0, 2.0]),
            y=np.array([1.0, -1.0, 0.0, 3.0]),
        )
        covariance = increment_covariance(
            trajectory.elapsed_seconds,
            sigma=3.0,
            location_error=2.0,
        )
        dense = -(
            gaussian_log_density(np.diff(trajectory.x), covariance)
            + gaussian_log_density(np.diff(trajectory.y), covariance)
        )
        banded = negative_log_likelihood(3.0, trajectory, 2.0)
        self.assertAlmostEqual(banded, dense, places=11)

    def test_joint_parameter_estimator_returns_finite_standard_deviations(self):
        rng = np.random.default_rng(52)
        count = 250
        true_sigma = 3.0
        true_error = 2.0
        brownian_x = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, true_sigma, count - 1))])
        brownian_y = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, true_sigma, count - 1))])
        trajectory = ProjectedTrajectory(
            identifier="joint",
            elapsed_seconds=np.arange(count, dtype=float),
            x=brownian_x + rng.normal(0.0, true_error, count),
            y=brownian_y + rng.normal(0.0, true_error, count),
        )
        estimate = estimate_motion_parameters(trajectory)
        self.assertAlmostEqual(estimate.sigma, true_sigma, delta=0.8)
        self.assertAlmostEqual(estimate.location_error, true_error, delta=0.8)


if __name__ == "__main__":
    unittest.main()
