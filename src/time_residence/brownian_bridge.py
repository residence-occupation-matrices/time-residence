"""Brownian-bridge parameter and occupation-time estimation.

The likelihood follows Appendix 2 of the associated article. ``sigma`` is the
Brownian-motion standard-deviation parameter: it enters the increment
covariance as ``Delta t * sigma**2``. The original production script applied
an additional square root after optimizing this parameter; this module does
not, because that changes its units and the subsequent bridge variance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.linalg import (
    cho_factor,
    cho_solve,
    cho_solve_banded,
    cholesky_banded,
)
from scipy.optimize import minimize, minimize_scalar

DEFAULT_LOCATION_ERROR_METERS: Final = 28.85
ARTICLE_PROJECTED_CRS: Final = "EPSG:32612"
RESEARCH_CODE_CRS: Final = "EPSG:3857"


@dataclass(frozen=True, slots=True)
class MotionParameters:
    """Brownian-motion and observation-error standard deviations, in metres."""

    sigma: float
    location_error: float

    def validate(self) -> None:
        values = np.asarray([self.sigma, self.location_error], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Motion parameters must be finite.")
        if self.sigma <= 0 or self.location_error < 0:
            raise ValueError("sigma must be positive and location_error non-negative.")


@dataclass(frozen=True, slots=True)
class ProjectedTrajectory:
    """Time-ordered locations in a projected coordinate system."""

    identifier: str
    elapsed_seconds: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def validate(self) -> None:
        n = len(self.elapsed_seconds)
        if n < 2:
            raise ValueError(
                f"Trajectory {self.identifier!r} has fewer than two unique timestamps."
            )
        if self.x.shape != (n,) or self.y.shape != (n,):
            raise ValueError(
                "Trajectory coordinate arrays must have the same one-dimensional shape."
            )
        if not np.all(np.isfinite(np.concatenate([self.elapsed_seconds, self.x, self.y]))):
            raise ValueError("Trajectory contains non-finite values.")
        if np.any(np.diff(self.elapsed_seconds) <= 0):
            raise ValueError("Trajectory timestamps must be strictly increasing.")


def project_trajectory(
    records: pd.DataFrame,
    *,
    projected_crs: str = ARTICLE_PROJECTED_CRS,
) -> ProjectedTrajectory:
    """Convert one device's longitude/latitude records to projected coordinates."""

    import geopandas as gpd

    required = {"id", "timestamp", "latitude", "longitude"}
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Trajectory records are missing columns: {missing}")
    identifiers = records["id"].astype(str).unique()
    if len(identifiers) != 1:
        raise ValueError("project_trajectory expects records from exactly one device.")

    ordered = records.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="raise")
    ordered["latitude"] = pd.to_numeric(ordered["latitude"], errors="raise")
    ordered["longitude"] = pd.to_numeric(ordered["longitude"], errors="raise")
    duplicate_times = ordered[ordered["timestamp"].duplicated(keep=False)]
    if not duplicate_times.empty:
        distinct_positions = duplicate_times.groupby("timestamp")[
            ["latitude", "longitude"]
        ].nunique()
        if (distinct_positions > 1).any(axis=None):
            raise ValueError(
                f"Trajectory {identifiers[0]!r} has conflicting positions at one timestamp."
            )
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    points = gpd.GeoDataFrame(
        ordered,
        geometry=gpd.points_from_xy(ordered["longitude"], ordered["latitude"]),
        crs="EPSG:4326",
    ).to_crs(projected_crs)
    if points.crs is None or points.crs.is_geographic:
        raise ValueError("projected_crs must identify a projected coordinate system.")
    elapsed = (points["timestamp"] - points["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    trajectory = ProjectedTrajectory(
        identifier=str(identifiers[0]),
        elapsed_seconds=elapsed.astype(float),
        x=points.geometry.x.to_numpy(dtype=float),
        y=points.geometry.y.to_numpy(dtype=float),
    )
    trajectory.validate()
    return trajectory


def increment_covariance(
    elapsed_seconds: np.ndarray,
    sigma: float,
    location_error: float,
) -> np.ndarray:
    """Covariance of consecutive observed displacements (article Eq. 9)."""

    elapsed = np.asarray(elapsed_seconds, dtype=float)
    dt = np.diff(elapsed)
    if dt.size == 0 or np.any(dt <= 0):
        raise ValueError("At least two strictly increasing observation times are required.")
    if sigma <= 0 or location_error < 0:
        raise ValueError("sigma must be positive and location_error must be non-negative.")

    diagonal = dt * sigma**2 + 2.0 * location_error**2
    covariance = np.diag(diagonal)
    if dt.size > 1:
        off_diagonal = np.full(dt.size - 1, -(location_error**2))
        covariance += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    return covariance


def _increment_covariance_banded(
    elapsed_seconds: np.ndarray,
    sigma: float,
    location_error: float,
) -> np.ndarray:
    """Return Eq. (9) in SciPy's lower-banded storage format."""

    elapsed = np.asarray(elapsed_seconds, dtype=float)
    dt = np.diff(elapsed)
    if dt.size == 0 or np.any(dt <= 0):
        raise ValueError("At least two strictly increasing observation times are required.")
    if sigma <= 0 or location_error < 0:
        raise ValueError("sigma must be positive and location_error must be non-negative.")
    covariance = np.zeros((2, dt.size), dtype=float)
    covariance[0] = dt * sigma**2 + 2.0 * location_error**2
    covariance[1, :-1] = -(location_error**2)
    return covariance


def gaussian_log_density(values: np.ndarray, covariance: np.ndarray) -> float:
    """Log density of a zero-mean multivariate normal distribution."""

    vector = np.asarray(values, dtype=float)
    factor, lower = cho_factor(covariance, lower=True, check_finite=True)
    solved = cho_solve((factor, lower), vector, check_finite=True)
    log_determinant_half = np.log(np.diag(factor)).sum()
    return float(
        -0.5 * vector.size * np.log(2.0 * np.pi) - log_determinant_half - 0.5 * vector.dot(solved)
    )


def negative_log_likelihood(
    sigma: float,
    trajectory: ProjectedTrajectory,
    location_error: float,
) -> float:
    """Negative joint log likelihood of x and y increments."""

    covariance = _increment_covariance_banded(
        trajectory.elapsed_seconds,
        sigma,
        location_error,
    )
    factor = cholesky_banded(covariance, lower=True, check_finite=True)
    log_determinant_half = np.log(factor[0]).sum()
    constant = -0.5 * (factor.shape[1] * np.log(2.0 * np.pi)) - log_determinant_half

    log_likelihood = 0.0
    for values in (np.diff(trajectory.x), np.diff(trajectory.y)):
        solved = cho_solve_banded((factor, True), values, check_finite=True)
        log_likelihood += constant - 0.5 * float(values.dot(solved))
    return -log_likelihood


def estimate_sigma(
    trajectory: ProjectedTrajectory,
    *,
    location_error: float = DEFAULT_LOCATION_ERROR_METERS,
    bounds: tuple[float, float] = (1.0e-6, 1.0e4),
) -> float:
    """Estimate the Brownian-motion standard deviation by maximum likelihood."""

    trajectory.validate()
    lower, upper = bounds
    if not 0 < lower < upper:
        raise ValueError("Sigma bounds must satisfy 0 < lower < upper.")

    result = minimize_scalar(
        lambda log_sigma: negative_log_likelihood(
            float(np.exp(log_sigma)), trajectory, location_error
        ),
        bounds=(float(np.log(lower)), float(np.log(upper))),
        method="bounded",
        options={"xatol": 1.0e-9},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Sigma optimization failed for device {trajectory.identifier!r}.")
    sigma = float(np.exp(result.x))
    if not lower <= sigma <= upper:
        raise RuntimeError(f"Estimated sigma {sigma} is outside the configured bounds.")
    return sigma


def estimate_motion_parameters(
    trajectory: ProjectedTrajectory,
    *,
    sigma_bounds: tuple[float, float] = (1.0e-6, 1.0e4),
    location_error_bounds: tuple[float, float] = (1.0e-6, 1.0e4),
) -> MotionParameters:
    """Jointly estimate ``sigma`` and location error using article Eq. (9).

    Optimization is performed in log space, so both standard deviations stay
    positive. This is the Pozdnyakov et al. formulation used when the location
    error is not known independently.
    """

    trajectory.validate()
    sigma_lower, sigma_upper = sigma_bounds
    error_lower, error_upper = location_error_bounds
    if not 0 < sigma_lower < sigma_upper:
        raise ValueError("Sigma bounds must satisfy 0 < lower < upper.")
    if not 0 < error_lower < error_upper:
        raise ValueError("Location-error bounds must satisfy 0 < lower < upper.")

    dt = np.diff(trajectory.elapsed_seconds)
    squared_displacement = np.diff(trajectory.x) ** 2 + np.diff(trajectory.y) ** 2
    sigma_start = float(np.sqrt(max(np.sum(squared_displacement) / (2.0 * np.sum(dt)), 1.0e-12)))
    sigma_start = float(np.clip(sigma_start, sigma_lower, sigma_upper))
    error_start = float(np.clip(DEFAULT_LOCATION_ERROR_METERS, error_lower, error_upper))

    def objective(log_parameters: np.ndarray) -> float:
        sigma, location_error = np.exp(log_parameters)
        return negative_log_likelihood(float(sigma), trajectory, float(location_error))

    result = minimize(
        objective,
        x0=np.log([sigma_start, error_start]),
        method="L-BFGS-B",
        bounds=[
            (float(np.log(sigma_lower)), float(np.log(sigma_upper))),
            (float(np.log(error_lower)), float(np.log(error_upper))),
        ],
        options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 1_000},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(
            f"Joint parameter optimization failed for device {trajectory.identifier!r}: "
            f"{result.message}"
        )
    parameters = MotionParameters(
        sigma=float(np.exp(result.x[0])),
        location_error=float(np.exp(result.x[1])),
    )
    parameters.validate()
    return parameters


def _contains_xy(geometry, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        from shapely import contains_xy

        return np.asarray(contains_xy(geometry, x, y), dtype=bool)
    except ImportError:  # Shapely 1.x compatibility for archived environments
        from shapely.geometry import Point

        return np.fromiter(
            (geometry.contains(Point(float(px), float(py))) for px, py in zip(x, y, strict=True)),
            dtype=bool,
            count=len(x),
        )


def patch_occupation_probability(
    trajectory: ProjectedTrajectory,
    geometry,
    sigma: float,
    *,
    location_error: float = DEFAULT_LOCATION_ERROR_METERS,
    samples: int = 1000,
    rng: np.random.Generator,
) -> float:
    """Estimate expected time in one patch by paired space-time Monte Carlo."""

    if samples < 1:
        raise ValueError("samples must be positive.")
    min_x, min_y, max_x, max_y = geometry.bounds
    area_box = (max_x - min_x) * (max_y - min_y)
    if not np.isfinite(area_box) or area_box <= 0:
        raise ValueError("Patch geometry has an invalid bounding box.")

    sample_x = rng.uniform(min_x, max_x, samples)
    sample_y = rng.uniform(min_y, max_y, samples)
    inside = _contains_xy(geometry, sample_x, sample_y).astype(float)
    if not inside.any():
        return 0.0

    total_duration = trajectory.elapsed_seconds[-1] - trajectory.elapsed_seconds[0]
    integral = 0.0
    for index, duration in enumerate(np.diff(trajectory.elapsed_seconds)):
        lower_time = np.nextafter(0.0, duration)
        upper_time = np.nextafter(duration, 0.0)
        relative_time = rng.uniform(lower_time, upper_time, samples)
        alpha = relative_time / duration
        mean_x = trajectory.x[index] + alpha * (trajectory.x[index + 1] - trajectory.x[index])
        mean_y = trajectory.y[index] + alpha * (trajectory.y[index + 1] - trajectory.y[index])
        variance = (
            relative_time * (1.0 - alpha) * sigma**2
            + ((1.0 - alpha) ** 2 + alpha**2) * location_error**2
        )
        normalizer = 1.0 / (2.0 * np.pi * variance)
        exponent = -((sample_x - mean_x) ** 2 + (sample_y - mean_y) ** 2) / (2.0 * variance)
        density = normalizer * np.exp(exponent)
        integral += duration * area_box * float(np.mean(inside * density))

    return integral / total_duration


def _trajectory_seed(base_seed: int, identifier: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{identifier}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def estimate_occupation_vector(
    trajectory: ProjectedTrajectory,
    agebs,
    sigma: float,
    *,
    location_error: float = DEFAULT_LOCATION_ERROR_METERS,
    samples: int = 1000,
    seed: int = 2025,
    projected_crs: str = ARTICLE_PROJECTED_CRS,
) -> np.ndarray:
    """Estimate and normalize one device's occupation probabilities by AGEB."""

    if agebs.crs is None:
        raise ValueError("AGEB geometries must declare a coordinate reference system.")
    projected = agebs.to_crs(projected_crs)
    if "ageb_index" not in projected.columns:
        raise ValueError("AGEB data must contain the stable ageb_index column.")
    projected = projected.sort_values("ageb_index")
    expected = np.arange(len(projected), dtype=int)
    if not np.array_equal(projected["ageb_index"].to_numpy(dtype=int), expected):
        raise ValueError("ageb_index must be contiguous and start at zero.")

    rng = np.random.default_rng(_trajectory_seed(seed, trajectory.identifier))
    probabilities = np.array(
        [
            patch_occupation_probability(
                trajectory,
                geometry,
                sigma,
                location_error=location_error,
                samples=samples,
                rng=rng,
            )
            for geometry in projected.geometry
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise RuntimeError(f"Invalid occupation estimate for device {trajectory.identifier!r}.")
    total = probabilities.sum()
    if total <= 0:
        raise RuntimeError(
            f"Occupation estimate for device {trajectory.identifier!r} has zero total mass."
        )
    return probabilities / total
