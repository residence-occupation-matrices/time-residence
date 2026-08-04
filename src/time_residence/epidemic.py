"""Multi-patch SEIRS simulation used for Figure 8 of the article."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from scipy.integrate import odeint

from .model_inputs import ModelInput, synthetic_model_input
from .periods import PERIOD_COMPARISONS

DEFAULT_SEED_AGEBS: Final[tuple[str, ...]] = ("2956", "3367", "5734", "6200")


@dataclass(frozen=True, slots=True)
class EpidemicParameters:
    """Homogeneous rates used in the published Figure 8 calculation.

    Section 5.3.3 reports beta, gamma, kappa, mu, and tau. The article defines
    the disease-induced mortality parameter but does not give its numerical
    value. The Figure 8 script uses ``phi=0.0003``; this implementation denotes
    the same parameter by ``psi``. All rates are per day.
    """

    beta: float = 1.5
    gamma: float = 1.0 / 14.0
    kappa: float = 1.0 / 7.0
    mu: float = 0.06 / (1000.0 * 365.0)
    psi: float = 0.0003
    tau: float = 1.0 / 180.0

    def validate(self) -> None:
        values = np.array(
            [self.beta, self.gamma, self.kappa, self.mu, self.psi, self.tau], dtype=float
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Epidemic rates must be finite and non-negative.")


DEFAULT_EPIDEMIC_PARAMETERS: Final = EpidemicParameters()


@dataclass(slots=True)
class PeriodComparison:
    """Infection differences for both panels of one Figure 8 row."""

    label: str
    time: np.ndarray
    ageb_ids: np.ndarray
    per_ageb_difference: np.ndarray
    global_difference: np.ndarray
    evaluation_day: float
    negative_fraction: float


def effective_rom(rom: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return ``P* = diag(alpha) P + diag(1-alpha)``."""

    matrix = np.asarray(rom, dtype=float)
    mobility = np.asarray(alpha, dtype=float).reshape(-1)
    if matrix.shape != (len(mobility), len(mobility)):
        raise ValueError("ROM and alpha dimensions do not agree.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(mobility)):
        raise ValueError("ROM and alpha must contain finite values.")
    if np.any(matrix < 0.0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1.0e-8, rtol=0.0):
        raise ValueError("ROM must be non-negative and row-stochastic.")
    if np.any(mobility < 0.0) or np.any(mobility > 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    result = matrix * mobility[:, None]
    result[np.diag_indices_from(result)] += 1.0 - mobility
    if not np.allclose(result.sum(axis=1), 1.0, atol=1.0e-8, rtol=0.0):
        raise ValueError("Effective ROM is not row-stochastic.")
    return result


def seirs_rhs(
    state: np.ndarray,
    _time: float,
    pstar: np.ndarray,
    population: np.ndarray,
    parameters: EpidemicParameters,
) -> np.ndarray:
    """Right-hand side of model (2), flattened as ``[S, E, I, R]``."""

    n = len(population)
    susceptible, exposed, infected, recovered = state.reshape(4, n)
    present_population = pstar.T @ population
    if np.any(present_population <= 0):
        raise ValueError("Effective population must be positive in every patch.")

    present_infected = pstar.T @ infected
    force_by_location = parameters.beta * present_infected / present_population
    new_exposures = susceptible * (pstar @ force_by_location)
    recruitment = parameters.mu * population

    d_susceptible = (
        recruitment - new_exposures - parameters.mu * susceptible + parameters.tau * recovered
    )
    d_exposed = new_exposures - (parameters.kappa + parameters.mu) * exposed
    d_infected = (
        parameters.kappa * exposed - (parameters.gamma + parameters.psi + parameters.mu) * infected
    )
    d_recovered = parameters.gamma * infected - (parameters.tau + parameters.mu) * recovered
    return np.concatenate([d_susceptible, d_exposed, d_infected, d_recovered])


def initial_state(
    model_input: ModelInput,
    seed_agebs: tuple[str, ...] | list[str] = DEFAULT_SEED_AGEBS,
    *,
    exposed_per_seed: float = 1.0,
    infected_per_seed: float = 1.0,
) -> np.ndarray:
    """Build the initial condition stated in Section 5.3.3."""

    identifiers = np.asarray(model_input.ageb_ids, dtype=str)
    requested = {str(code) for code in seed_agebs}
    if not requested:
        raise ValueError("At least one seed AGEB is required.")
    missing = requested - set(identifiers)
    if missing:
        raise ValueError(f"Seed AGEBs {sorted(missing)} do not occur in {model_input.period}.")
    selected = np.isin(identifiers, list(requested))

    exposed = np.zeros(model_input.n_patches, dtype=float)
    infected = np.zeros(model_input.n_patches, dtype=float)
    recovered = np.zeros(model_input.n_patches, dtype=float)
    exposed[selected] = exposed_per_seed
    infected[selected] = infected_per_seed
    susceptible = np.asarray(model_input.population, dtype=float) - exposed - infected
    if np.any(susceptible < 0):
        raise ValueError("Initial exposed and infected counts exceed a patch population.")
    return np.concatenate([susceptible, exposed, infected, recovered])


def simulate_infected(
    model_input: ModelInput,
    time: np.ndarray,
    *,
    parameters: EpidemicParameters = DEFAULT_EPIDEMIC_PARAMETERS,
    seed_agebs: tuple[str, ...] | list[str] = DEFAULT_SEED_AGEBS,
) -> np.ndarray:
    """Integrate one period-part and return ``I_i(t)`` for every AGEB."""

    model_input.validate()
    parameters.validate()
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("Simulation times must be a strictly increasing one-dimensional array.")
    pstar = effective_rom(model_input.rom, model_input.alpha)
    population = np.asarray(model_input.population, dtype=float).reshape(-1)
    state_zero = initial_state(model_input, seed_agebs)
    solution, diagnostics = odeint(
        seirs_rhs,
        state_zero,
        time,
        args=(pstar, population, parameters),
        rtol=1.0e-9,
        atol=1.0e-9,
        mxstep=10_000,
        full_output=True,
    )
    if diagnostics.get("message") != "Integration successful.":
        raise RuntimeError(
            f"Integration for {model_input.period} failed: "
            f"{diagnostics.get('message', 'unknown solver status')}"
        )
    if not np.all(np.isfinite(solution)):
        raise RuntimeError(f"Integration for {model_input.period} produced non-finite values.")
    infected = solution[:, 2 * model_input.n_patches : 3 * model_input.n_patches]
    if np.min(infected) < -1.0e-6:
        raise RuntimeError(f"Integration for {model_input.period} produced negative infections.")
    return np.maximum(infected, 0.0)


def _common_indices(
    first: ModelInput,
    second: ModelInput,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_positions = {str(code): index for index, code in enumerate(first.ageb_ids)}
    second_positions = {str(code): index for index, code in enumerate(second.ageb_ids)}
    common = np.array(sorted(first_positions.keys() & second_positions.keys()), dtype=str)
    if common.size == 0:
        raise ValueError(f"{first.period} and {second.period} have no common AGEBs.")
    first_index = np.array([first_positions[code] for code in common], dtype=int)
    second_index = np.array([second_positions[code] for code in common], dtype=int)
    return common, first_index, second_index


def compare_period(
    label: str,
    first: ModelInput,
    second: ModelInput,
    time: np.ndarray,
    *,
    parameters: EpidemicParameters = DEFAULT_EPIDEMIC_PARAMETERS,
    seed_agebs: tuple[str, ...] | list[str] = DEFAULT_SEED_AGEBS,
    evaluation_day: float = 30.0,
) -> PeriodComparison:
    """Compute first-part minus second-part infections for one period."""

    first_infected = simulate_infected(first, time, parameters=parameters, seed_agebs=seed_agebs)
    second_infected = simulate_infected(second, time, parameters=parameters, seed_agebs=seed_agebs)
    common, first_index, second_index = _common_indices(first, second)
    difference = first_infected[:, first_index] - second_infected[:, second_index]
    global_difference = difference.sum(axis=1)
    evaluation_index = int(np.argmin(np.abs(np.asarray(time) - evaluation_day)))
    actual_day = float(np.asarray(time)[evaluation_index])
    negative_fraction = float(np.mean(difference[evaluation_index] < 0.0))
    return PeriodComparison(
        label=label,
        time=np.asarray(time, dtype=float),
        ageb_ids=common,
        per_ageb_difference=difference,
        global_difference=global_difference,
        evaluation_day=actual_day,
        negative_fraction=negative_fraction,
    )


def plot_figure8(comparisons: list[PeriodComparison], path: str | Path, *, dpi: int = 300) -> Path:
    """Render the six panels corresponding to Figure 8."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True, squeeze=False)
    panel_labels = ("A", "B", "C", "D", "E", "F")
    for row, comparison in enumerate(comparisons):
        local_axis, global_axis = axes[row]
        local_axis.plot(
            comparison.time,
            comparison.per_ageb_difference,
            linewidth=0.55,
            alpha=0.75,
        )
        global_axis.plot(
            comparison.time,
            comparison.global_difference,
            color="black",
            linewidth=1.4,
        )
        for column, axis in enumerate((local_axis, global_axis)):
            panel = panel_labels[2 * row + column]
            axis.text(
                0.01,
                0.98,
                panel,
                transform=axis.transAxes,
                ha="left",
                va="top",
                weight="bold",
            )
            axis.axhline(0.0, color="0.55", linewidth=0.7)
            axis.grid(True, color="0.9", linewidth=0.6)
            axis.set_xlabel("Time (days)")
            axis.set_ylabel("Difference in infected individuals")
        local_axis.set_title(f"{comparison.label}: individual AGEBs")
        global_axis.set_title(f"{comparison.label}: citywide sum")

    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def save_comparisons(comparisons: list[PeriodComparison], output_dir: str | Path) -> None:
    """Store complete numeric panels and a compact JSON summary."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float | int]] = {}
    for comparison in comparisons:
        np.savez_compressed(
            destination / f"figure8_{comparison.label}.npz",
            time=comparison.time,
            ageb_ids=comparison.ageb_ids,
            per_ageb_difference=comparison.per_ageb_difference,
            global_difference=comparison.global_difference,
        )
        summary[comparison.label] = {
            "common_agebs": int(len(comparison.ageb_ids)),
            "evaluation_day": comparison.evaluation_day,
            "negative_fraction": comparison.negative_fraction,
        }
    (destination / "figure8_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def reproduce_figure8(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    parameters: EpidemicParameters = DEFAULT_EPIDEMIC_PARAMETERS,
    seed_agebs: tuple[str, ...] | list[str] = DEFAULT_SEED_AGEBS,
    time_end: float = 200.0,
    time_points: int = 100,
    evaluation_day: float = 30.0,
) -> list[PeriodComparison]:
    """Load all six period-parts, simulate them, and save Figure 8 outputs."""

    if time_end <= 0 or time_points < 2:
        raise ValueError("time_end must be positive and time_points must be at least two.")
    source = Path(input_dir)
    time = np.linspace(0.0, time_end, time_points)
    comparisons: list[PeriodComparison] = []
    for label, (first_code, second_code) in PERIOD_COMPARISONS.items():
        first = ModelInput.load(source / f"{first_code}.npz")
        second = ModelInput.load(source / f"{second_code}.npz")
        comparisons.append(
            compare_period(
                label,
                first,
                second,
                time,
                parameters=parameters,
                seed_agebs=seed_agebs,
                evaluation_day=evaluation_day,
            )
        )

    destination = Path(output_dir)
    plot_figure8(comparisons, destination / "figure8.png")
    save_comparisons(comparisons, destination)
    return comparisons


def write_synthetic_figure8_inputs(
    output_dir: str | Path,
    *,
    n_patches: int = 12,
    seed: int = 2025,
) -> tuple[str, ...]:
    """Write six deterministic inputs sharing a valid synthetic seed AGEB."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for offset, code in enumerate(code for pair in PERIOD_COMPARISONS.values() for code in pair):
        item = synthetic_model_input(code, n_patches=n_patches, seed=seed + offset)
        item.ageb_ids = np.array([f"A{index:04d}" for index in range(n_patches)])
        item.save(destination / f"{code}.npz")
    return ("A0000",)
