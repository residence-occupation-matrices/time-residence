"""Preparation of GPS records and residence assignments."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Final

import numpy as np
import pandas as pd

from .periods import STUDY_PERIODS, night_mask

PING_COLUMNS: Final = ("id", "timestamp", "latitude", "longitude")
RESEARCH_COLUMN_NAMES: Final = {
    "id_adv": "id",
    "lat": "latitude",
    "lon": "longitude",
    "polygon": "ageb_index",
}


def normalize_ping_columns(
    frame: pd.DataFrame, *, repair_swapped_coordinates: bool = False
) -> pd.DataFrame:
    """Return a copy with the canonical ping column names.

    The first repository version wrote latitude and longitude values under
    interchanged CSV headers. When that signature is detected, the function
    raises unless ``repair_swapped_coordinates`` is explicitly enabled.
    """

    rename = {
        old: new
        for old, new in RESEARCH_COLUMN_NAMES.items()
        if old in frame.columns and new not in frame.columns
    }
    result = frame.rename(columns=rename).copy()
    missing = [name for name in PING_COLUMNS if name not in result.columns]
    if missing:
        raise ValueError(f"Ping table is missing required columns: {missing}")

    latitude = pd.to_numeric(result["latitude"], errors="coerce")
    longitude = pd.to_numeric(result["longitude"], errors="coerce")
    if latitude.isna().any() or longitude.isna().any():
        raise ValueError("Latitude and longitude must be finite numeric values.")

    latitude_valid = latitude.between(-90.0, 90.0).all()
    longitude_valid = longitude.between(-180.0, 180.0).all()
    swapped_signature = (
        not latitude_valid
        and longitude.between(-90.0, 90.0).all()
        and latitude.between(-180.0, 180.0).all()
    )
    if swapped_signature:
        if not repair_swapped_coordinates:
            raise ValueError(
                "The coordinate columns have the signature of the latitude/longitude "
                "header inversion in the original calculate-ageb.py output. Re-run with "
                "repair_swapped_coordinates=True only after confirming the source file."
            )
        latitude, longitude = longitude, latitude
        latitude_valid = longitude_valid = True

    if not latitude_valid or not longitude_valid:
        raise ValueError("Coordinates fall outside valid latitude/longitude ranges.")

    identifiers = result["id"]
    if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
        raise ValueError("Device identifiers must be non-empty.")
    result["latitude"] = latitude.astype(float)
    result["longitude"] = longitude.astype(float)
    result["id"] = identifiers.astype(str)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    return result


def prepare_agebs(agebs, *, ageb_code_column: str = "CVE_AGEB"):
    """Return AGEB geometries with a stable, zero-based matrix index."""

    if ageb_code_column not in agebs.columns:
        raise ValueError(f"AGEB data is missing '{ageb_code_column}'.")
    if "geometry" not in agebs.columns:
        raise ValueError("AGEB data is missing geometry.")
    if agebs.crs is None:
        raise ValueError("AGEB geometries must declare a coordinate reference system.")
    result = agebs.copy()
    codes = result[ageb_code_column]
    if codes.isna().any() or codes.astype(str).str.strip().eq("").any():
        raise ValueError("AGEB codes must be non-empty.")
    if result["geometry"].isna().any():
        raise ValueError("AGEB geometries must be non-empty.")
    empty_geometry = result["geometry"].map(lambda geometry: geometry.is_empty)
    if empty_geometry.any():
        raise ValueError("AGEB geometries must be non-empty.")
    result[ageb_code_column] = codes.astype(str)
    if result[ageb_code_column].duplicated().any():
        raise ValueError(f"AGEB codes in '{ageb_code_column}' must be unique.")
    result = result.sort_values(ageb_code_column).reset_index(drop=True)
    result["ageb_index"] = np.arange(len(result), dtype=int)
    return result


def assign_agebs(
    pings: pd.DataFrame,
    agebs,
    *,
    ageb_code_column: str = "CVE_AGEB",
    predicate: str = "within",
) -> pd.DataFrame:
    """Assign each ping to an AGEB with a spatial-indexed join.

    ``agebs`` must be a GeoDataFrame with a defined coordinate reference
    system. Rows are sorted by AGEB code before assigning ``ageb_index`` so the
    index is stable across all pipeline stages.
    """

    import geopandas as gpd

    pings = normalize_ping_columns(pings).drop(
        columns=["ageb_index", "CVE_AGEB"],
        errors="ignore",
    )
    ageb_table = prepare_agebs(agebs, ageb_code_column=ageb_code_column)[
        [ageb_code_column, "ageb_index", "geometry"]
    ]

    points = gpd.GeoDataFrame(
        pings,
        geometry=gpd.points_from_xy(pings["longitude"], pings["latitude"]),
        crs="EPSG:4326",
    ).to_crs(ageb_table.crs)
    joined = gpd.sjoin(
        points,
        ageb_table[[ageb_code_column, "ageb_index", "geometry"]],
        how="left",
        predicate=predicate,
    )
    if len(joined) != len(points):
        raise ValueError("Spatial join produced duplicate matches; AGEB polygons may overlap.")

    joined["ageb_index"] = joined["ageb_index"].fillna(-1).astype(int)
    joined = joined.rename(columns={ageb_code_column: "CVE_AGEB"})
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def select_period(
    pings: pd.DataFrame,
    period_code: str,
    *,
    eligible_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Select one published period using local, inclusive calendar dates."""

    if period_code not in STUDY_PERIODS:
        raise ValueError(f"Unknown period '{period_code}'. Choose from {sorted(STUDY_PERIODS)}.")
    result = normalize_ping_columns(pings)
    if eligible_ids is not None:
        allowed = {str(value) for value in eligible_ids}
        result = result[result["id"].isin(allowed)]
    return result.loc[STUDY_PERIODS[period_code].mask(result["timestamp"])].copy()


def select_night_records(pings: pd.DataFrame) -> pd.DataFrame:
    """Return records observed from 22:00 through, but not including, 06:00."""

    return pings.loc[night_mask(pings["timestamp"])].copy()


def _modes(values: pd.Series) -> list[int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    clean = clean[clean >= 0]
    return sorted(clean.mode().tolist())


def _id_seed(base_seed: int, identifier: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{identifier}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def infer_residences(
    pings: pd.DataFrame,
    ageb_metadata: pd.DataFrame,
    *,
    population_column: str = "POBTOT",
    seed: int = 2025,
) -> pd.DataFrame:
    """Assign one residence AGEB to each device.

    Candidates are modal AGEBs in both the full and night-time records. If the
    modal sets do not intersect, night-time modes take precedence. Ties are
    sampled in proportion to census population. A device-specific random seed
    makes the result independent of row order and worker scheduling.
    """

    required = {"ageb_index", "CVE_AGEB", population_column}
    missing = sorted(required - set(ageb_metadata.columns))
    if missing:
        raise ValueError(f"AGEB metadata is missing columns: {missing}")
    if "ageb_index" not in pings.columns:
        raise ValueError("Pings must include ageb_index; run assign-agebs first.")

    metadata = ageb_metadata.copy()
    metadata["ageb_index"] = metadata["ageb_index"].astype(int)
    metadata["CVE_AGEB"] = metadata["CVE_AGEB"].astype(str)
    population = metadata.set_index("ageb_index")[population_column].astype(float).to_dict()
    code = metadata.set_index("ageb_index")["CVE_AGEB"].to_dict()

    records = pings.copy()
    records["id"] = records["id"].astype(str)
    records["_is_night"] = night_mask(records["timestamp"]).to_numpy(dtype=bool)
    rows: list[dict[str, object]] = []

    for identifier, group in records.groupby("id", sort=True):
        all_modes = _modes(group["ageb_index"])
        night_group = group.loc[group["_is_night"]]
        night_modes = _modes(night_group["ageb_index"])
        candidates = sorted(set(all_modes).intersection(night_modes))
        if not candidates:
            candidates = night_modes or all_modes

        if not candidates:
            residence = -1
        elif len(candidates) == 1:
            residence = candidates[0]
        else:
            weights = np.array([max(population.get(item, 0.0), 0.0) for item in candidates])
            probabilities = None if weights.sum() == 0 else weights / weights.sum()
            rng = np.random.default_rng(_id_seed(seed, identifier))
            residence = int(rng.choice(candidates, p=probabilities))

        rows.append(
            {
                "id": identifier,
                "residence_ageb_index": residence,
                "residence_CVE_AGEB": code.get(residence, pd.NA),
                "all_record_modes": ",".join(map(str, all_modes)),
                "night_record_modes": ",".join(map(str, night_modes)),
            }
        )

    return pd.DataFrame(rows)


def classify_leavers(pings: pd.DataFrame, residences: pd.DataFrame) -> pd.DataFrame:
    """Report whether each resident has at least one ping outside the home AGEB."""

    required = {"id", "ageb_index"}
    if not required.issubset(pings.columns):
        raise ValueError(f"Pings must contain {sorted(required)}.")
    if not {"id", "residence_ageb_index"}.issubset(residences.columns):
        raise ValueError("Residences must contain id and residence_ageb_index.")

    merged = pings[["id", "ageb_index"]].copy()
    merged["id"] = merged["id"].astype(str)
    homes = residences[["id", "residence_ageb_index"]].copy()
    homes["id"] = homes["id"].astype(str)
    merged = merged.merge(homes, on="id", how="inner", validate="many_to_one")
    observed = pd.to_numeric(merged["ageb_index"], errors="coerce")
    residence = pd.to_numeric(merged["residence_ageb_index"], errors="coerce")
    valid = observed.notna() & residence.notna() & (observed >= 0) & (residence >= 0)
    merged["outside_home"] = valid & (observed != residence)
    result = merged.groupby("id", sort=True)["outside_home"].any().rename("leaves_home")
    return result.reset_index()
