"""Construction and validation of epidemic-model inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .periods import STUDY_PERIODS
from .preprocessing import classify_leavers


@dataclass(slots=True)
class ModelInput:
    """Aligned mobility, population, and patch-identification arrays."""

    period: str
    ageb_ids: np.ndarray
    rom: np.ndarray
    alpha: np.ndarray
    population: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_patches(self) -> int:
        return int(np.asarray(self.rom).shape[0])

    def validate(self, *, tolerance: float = 1.0e-8) -> None:
        """Reject misalignment and values outside the mathematical contract."""

        rom = np.asarray(self.rom, dtype=float)
        alpha = np.asarray(self.alpha, dtype=float).reshape(-1)
        population = np.asarray(self.population, dtype=float).reshape(-1)
        n = int(rom.shape[0]) if rom.ndim == 2 else 0
        if rom.shape != (n, n):
            raise ValueError(f"ROM must be square; received {rom.shape}.")
        if np.asarray(self.ageb_ids).shape != (n,):
            raise ValueError("ageb_ids must contain one identifier per ROM row.")
        if alpha.shape != (n,):
            raise ValueError("alpha must contain one value per ROM row.")
        if population.shape != (n,):
            raise ValueError("population must contain one value per ROM row.")
        if len(set(map(str, self.ageb_ids))) != n:
            raise ValueError("AGEB identifiers must be unique.")

        arrays = (rom, alpha, population)
        if not all(np.all(np.isfinite(np.asarray(array, dtype=float))) for array in arrays):
            raise ValueError("Model input contains non-finite numeric values.")
        if np.any(rom < 0.0):
            raise ValueError("ROM entries must be non-negative.")
        if not np.allclose(rom.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
            raise ValueError("Every ROM row must sum to one.")
        if np.any(alpha < 0.0) or np.any(alpha > 1.0):
            raise ValueError("Mobility fractions alpha must lie in [0, 1].")
        if np.any(population <= 0):
            raise ValueError("Patch populations must be strictly positive.")

    def save(self, path: str | Path) -> Path:
        """Write a compressed, alignment-preserving NPZ artifact."""

        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "period": self.period}
        np.savez_compressed(
            output,
            ageb_ids=np.asarray(self.ageb_ids, dtype=str),
            rom=np.asarray(self.rom, dtype=float),
            alpha=np.asarray(self.alpha, dtype=float).reshape(-1),
            population=np.asarray(self.population, dtype=float).reshape(-1),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> ModelInput:
        """Load and validate a model-input artifact without enabling pickle."""

        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            required = {"ageb_ids", "rom", "alpha", "population", "metadata_json"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"{source} is missing arrays: {missing}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            period = str(metadata.pop("period", source.stem))
            result = cls(
                period=period,
                ageb_ids=archive["ageb_ids"].astype(str),
                rom=archive["rom"].astype(float),
                alpha=archive["alpha"].astype(float),
                population=archive["population"].astype(float),
                metadata=metadata,
            )
        result.validate()
        return result


def load_individual_roms(path: str | Path) -> dict[str, np.ndarray]:
    """Load the research-script JSON mapping of devices to occupation vectors."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Individual ROM JSON must be an object keyed by device id.")
    return {str(key): np.asarray(value, dtype=float) for key, value in payload.items()}


def build_model_input(
    period: str,
    individual_roms: dict[str, np.ndarray],
    pings: pd.DataFrame,
    residences: pd.DataFrame,
    ageb_metadata: pd.DataFrame,
    *,
    population_column: str = "POBTOT",
) -> ModelInput:
    """Estimate alpha and the ROM conditioned on residents who leave home.

    ``alpha_i`` is the fraction of sampled residents of patch ``i`` with at
    least one observed ping outside ``i``. Row ``i`` of ``rom`` averages only
    those residents' normalized occupation vectors, as required by model (2)
    in the article. If ``alpha_i = 0``, the otherwise unused conditional row is
    set to the identity row so the matrix remains stochastic.
    """

    if period not in STUDY_PERIODS:
        raise ValueError(f"Unknown period {period!r}; expected one of {sorted(STUDY_PERIODS)}.")

    metadata_required = {"ageb_index", "CVE_AGEB", population_column}
    missing = sorted(metadata_required - set(ageb_metadata.columns))
    if missing:
        raise ValueError(f"AGEB metadata is missing columns: {missing}")

    agebs = ageb_metadata.copy()
    agebs["ageb_index"] = agebs["ageb_index"].astype(int)
    agebs["CVE_AGEB"] = agebs["CVE_AGEB"].astype(str)
    agebs = agebs.sort_values("ageb_index").reset_index(drop=True)
    expected = np.arange(len(agebs), dtype=int)
    if not np.array_equal(agebs["ageb_index"].to_numpy(), expected):
        raise ValueError("ageb_index must be contiguous and start at zero.")

    homes = residences[["id", "residence_ageb_index"]].copy()
    homes["id"] = homes["id"].astype(str)
    homes["residence_ageb_index"] = homes["residence_ageb_index"].astype(int)
    homes = homes[homes["residence_ageb_index"] >= 0]
    if homes["id"].duplicated().any():
        raise ValueError("Residence table contains duplicate device identifiers.")
    if homes.empty:
        raise ValueError("No valid residence assignments are available.")
    observed_ids = set(pings["id"].astype(str))
    missing_pings = set(homes["id"]) - observed_ids
    if missing_pings:
        raise ValueError(f"Residence table contains {len(missing_pings)} devices without pings.")

    leaving = classify_leavers(pings, homes)
    residents = homes.merge(leaving, on="id", how="left", validate="one_to_one")
    residents["leaves_home"] = residents["leaves_home"].fillna(False).astype(bool)
    kept_indices = np.sort(residents["residence_ageb_index"].unique())
    if np.any(kept_indices >= len(agebs)):
        raise ValueError("Residence table refers to an ageb_index outside the metadata table.")

    n_full = len(agebs)
    n_kept = len(kept_indices)
    rom = np.zeros((n_kept, n_kept), dtype=float)
    alpha = np.zeros(n_kept, dtype=float)
    resident_counts = np.zeros(n_kept, dtype=int)
    leaver_counts = np.zeros(n_kept, dtype=int)

    for row_index, home_index in enumerate(kept_indices):
        group = residents[residents["residence_ageb_index"] == home_index]
        resident_counts[row_index] = len(group)
        leavers = group[group["leaves_home"]]
        leaver_counts[row_index] = len(leavers)
        alpha[row_index] = len(leavers) / len(group)

        if leavers.empty:
            rom[row_index, row_index] = 1.0
            continue

        vectors: list[np.ndarray] = []
        for identifier in leavers["id"]:
            if identifier not in individual_roms:
                raise ValueError(f"Missing individual ROM for leaving device {identifier!r}.")
            vector = np.asarray(individual_roms[identifier], dtype=float).reshape(-1)
            if vector.shape != (n_full,):
                raise ValueError(
                    f"Individual ROM for {identifier!r} has length {len(vector)}; "
                    f"expected {n_full}."
                )
            if not np.all(np.isfinite(vector)) or np.any(vector < 0):
                raise ValueError(f"Individual ROM for {identifier!r} contains invalid values.")
            restricted = vector[kept_indices]
            total = restricted.sum()
            if total <= 0:
                raise ValueError(
                    f"Individual ROM for {identifier!r} has no mass in retained AGEBs."
                )
            vectors.append(restricted / total)

        rom[row_index] = np.mean(vectors, axis=0)
        rom[row_index] /= rom[row_index].sum()

    selected = agebs.set_index("ageb_index").loc[kept_indices]
    population = pd.to_numeric(selected[population_column], errors="coerce").to_numpy(float)
    result = ModelInput(
        period=period,
        ageb_ids=selected["CVE_AGEB"].to_numpy(str),
        rom=rom,
        alpha=alpha,
        population=population,
        metadata={
            "rom_conditioning": "residents with at least one observed ping outside home AGEB",
            "alpha_definition": (
                "fraction of sampled residents with an observed ping outside home AGEB"
            ),
            "resident_counts": resident_counts.tolist(),
            "leaver_counts": leaver_counts.tolist(),
        },
    )
    result.validate()
    return result


def import_matrix_csv_input(
    period: str,
    matrix_path: str | Path,
    alpha_population_csv: str | Path,
) -> ModelInput:
    """Import one aligned matrix/CSV pair used by the Figure 8 script."""

    if period not in STUDY_PERIODS:
        raise ValueError(f"Unknown period {period!r}; expected one of {sorted(STUDY_PERIODS)}.")

    matrix = np.load(Path(matrix_path), allow_pickle=False).astype(float)
    frame = pd.read_csv(
        alpha_population_csv,
        dtype={"CVE_AGEB": "string"},
    ).dropna(subset=["proporcion"])
    required = {"CVE_AGEB", "proporcion", "POBTOT"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Akuno input CSV is missing columns: {missing}")
    if matrix.shape != (len(frame), len(frame)):
        raise ValueError(f"Matrix shape {matrix.shape} does not align with {len(frame)} CSV rows.")

    result = ModelInput(
        period=period,
        ageb_ids=frame["CVE_AGEB"].astype(str).to_numpy(),
        rom=matrix,
        alpha=1.0 - pd.to_numeric(frame["proporcion"]).to_numpy(float),
        population=pd.to_numeric(frame["POBTOT"]).to_numpy(float),
        metadata={
            "source": "research/scripts/epidemic_simulation_code.py input contract",
            "proporcion_definition": "fraction of residents who do not leave home AGEB",
        },
    )
    result.validate()
    return result


def synthetic_model_input(
    period: str,
    *,
    n_patches: int = 12,
    seed: int = 2025,
) -> ModelInput:
    """Generate a deterministic model input for examples and tests."""

    if n_patches < 2:
        raise ValueError("Synthetic input requires at least two patches.")
    rng = np.random.default_rng(seed)
    matrix = rng.random((n_patches, n_patches))
    matrix /= matrix.sum(axis=1, keepdims=True)
    result = ModelInput(
        period=period,
        ageb_ids=np.array([f"A{i:04d}" for i in range(n_patches)]),
        rom=matrix,
        alpha=rng.uniform(0.1, 0.8, n_patches),
        population=rng.integers(500, 15_000, n_patches).astype(float),
        metadata={"source": "deterministic synthetic example", "seed": seed},
    )
    result.validate()
    return result
