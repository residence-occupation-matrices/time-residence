"""Command-line interface for data preparation and epidemic simulation."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .brownian_bridge import (
    ARTICLE_PROJECTED_CRS,
    DEFAULT_LOCATION_ERROR_METERS,
    MotionParameters,
    estimate_motion_parameters,
    estimate_occupation_vector,
    estimate_sigma,
    project_trajectory,
)
from .epidemic import (
    DEFAULT_SEED_AGEBS,
    EpidemicParameters,
    reproduce_figure8,
    write_synthetic_figure8_inputs,
)
from .model_inputs import (
    build_model_input,
    import_matrix_csv_input,
    load_individual_roms,
)
from .periods import STUDY_PERIODS
from .preprocessing import (
    assign_agebs,
    infer_residences,
    normalize_ping_columns,
    prepare_agebs,
    select_period,
)

LOGGER = logging.getLogger("time_residence")

ARTICLE_INPUT_FILES = {
    "P1A": ("working_avg_resmat_Second_First.npy", "alphas_SF.csv"),
    "P1B": ("working_avg_resmat_First_Second.npy", "alphas_FS.csv"),
    "P2A": ("working_avg_resmat_Second_First.npy", "alphas_SF.csv"),
    "P2B": ("working_avg_resmat_Second_Second.npy", "alphas_SS.csv"),
    "P3A": ("working_avg_resmat_Third_First.npy", "alphas_TF.csv"),
    "P3B": ("working_avg_resmat_Third_Second.npy", "alphas_TS.csv"),
}


def _read_csv(path: Path, separator: str, *, repair_swapped: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=separator)
    return normalize_ping_columns(frame, repair_swapped_coordinates=repair_swapped)


def _write_csv(frame: pd.DataFrame, path: Path, separator: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=separator, index=False)


def _add_separator(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--separator",
        default=",",
        help="CSV delimiter for input and output files (default: comma)",
    )


def _command_assign_agebs(args: argparse.Namespace) -> int:
    import geopandas as gpd

    pings = _read_csv(args.pings, args.separator, repair_swapped=args.repair_swapped)
    agebs = gpd.read_file(args.agebs)
    assigned = assign_agebs(pings, agebs, ageb_code_column=args.ageb_code_column)
    _write_csv(assigned, args.output, args.separator)
    LOGGER.info(
        "Assigned %d pings; %d fall outside the AGEB polygons.",
        len(assigned),
        (assigned["ageb_index"] < 0).sum(),
    )
    return 0


def _command_prepare_agebs(args: argparse.Namespace) -> int:
    import geopandas as gpd

    agebs = prepare_agebs(
        gpd.read_file(args.agebs),
        ageb_code_column=args.ageb_code_column,
    )
    if args.population_column not in agebs.columns:
        raise ValueError(f"AGEB data is missing '{args.population_column}'.")
    metadata = agebs[["ageb_index", args.ageb_code_column, args.population_column]].rename(
        columns={args.ageb_code_column: "CVE_AGEB"}
    )
    metadata[args.population_column] = pd.to_numeric(
        metadata[args.population_column], errors="raise"
    )
    _write_csv(metadata, args.output, args.separator)
    LOGGER.info("Wrote aligned metadata for %d AGEBs.", len(metadata))
    return 0


def _command_select_period(args: argparse.Namespace) -> int:
    pings = _read_csv(args.pings, args.separator, repair_swapped=args.repair_swapped)
    eligible = None
    if args.eligible_ids:
        eligible_frame = pd.read_csv(args.eligible_ids, sep=args.separator, dtype=str)
        if args.id_column not in eligible_frame.columns:
            raise ValueError(f"Eligible-ID table is missing '{args.id_column}'.")
        eligible = eligible_frame[args.id_column].astype(str)
    selected = select_period(pings, args.period, eligible_ids=eligible)
    _write_csv(selected, args.output, args.separator)
    LOGGER.info("Selected %d pings for %s.", len(selected), args.period)
    return 0


def _command_assign_residences(args: argparse.Namespace) -> int:
    pings = _read_csv(args.pings, args.separator, repair_swapped=args.repair_swapped)
    metadata = pd.read_csv(
        args.ageb_metadata,
        sep=args.separator,
        dtype={"CVE_AGEB": "string"},
    )
    residences = infer_residences(
        pings,
        metadata,
        population_column=args.population_column,
        seed=args.seed,
    )
    _write_csv(residences, args.output, args.separator)
    unassigned = int((residences["residence_ageb_index"] < 0).sum())
    LOGGER.info(
        "Assigned %d residences; %d devices remain unassigned.",
        len(residences),
        unassigned,
    )
    return 0


def _command_estimate_individual_roms(args: argparse.Namespace) -> int:
    import geopandas as gpd
    from tqdm import tqdm

    pings = _read_csv(args.pings, args.separator, repair_swapped=args.repair_swapped)
    agebs = prepare_agebs(
        gpd.read_file(args.agebs),
        ageb_code_column=args.ageb_code_column,
    )
    occupation: dict[str, list[float]] = {}
    parameter_values: dict[str, dict[str, float]] = {}
    failures: dict[str, str] = {}

    groups = pings.groupby("id", sort=True)
    for identifier, records in tqdm(groups, total=pings["id"].nunique(), desc="Devices"):
        try:
            trajectory = project_trajectory(records, projected_crs=args.projected_crs)
            if args.estimate_location_error:
                parameters = estimate_motion_parameters(trajectory)
            else:
                parameters = MotionParameters(
                    sigma=estimate_sigma(
                        trajectory,
                        location_error=args.location_error,
                    ),
                    location_error=args.location_error,
                )
            vector = estimate_occupation_vector(
                trajectory,
                agebs,
                parameters.sigma,
                location_error=parameters.location_error,
                samples=args.samples,
                seed=args.seed,
                projected_crs=args.projected_crs,
            )
            parameter_values[str(identifier)] = {
                "sigma": parameters.sigma,
                "location_error": parameters.location_error,
            }
            occupation[str(identifier)] = vector.tolist()
        except (ValueError, RuntimeError) as error:
            failures[str(identifier)] = str(error)
            if args.fail_fast:
                raise

    if not occupation:
        raise RuntimeError("No device occupation vectors were estimated successfully.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(occupation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.parameter_output.parent.mkdir(parents=True, exist_ok=True)
    args.parameter_output.write_text(
        json.dumps(parameter_values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.failure_output.parent.mkdir(parents=True, exist_ok=True)
    args.failure_output.write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        LOGGER.warning("Skipped %d devices; details are in %s.", len(failures), args.failure_output)
    LOGGER.info("Estimated occupation vectors for %d devices.", len(occupation))
    return 0


def _command_build_model_input(args: argparse.Namespace) -> int:
    pings = _read_csv(args.pings, args.separator, repair_swapped=args.repair_swapped)
    residences = pd.read_csv(args.residences, sep=args.separator, dtype={"id": str})
    metadata = pd.read_csv(
        args.ageb_metadata,
        sep=args.separator,
        dtype={"CVE_AGEB": "string"},
    )
    individual_roms = load_individual_roms(args.individual_roms)
    result = build_model_input(
        args.period,
        individual_roms,
        pings,
        residences,
        metadata,
        population_column=args.population_column,
    )
    result.save(args.output)
    LOGGER.info("Wrote %s with %d aligned AGEBs.", args.output, result.n_patches)
    return 0


def _command_import_article_inputs(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for period, (matrix_name, alpha_name) in ARTICLE_INPUT_FILES.items():
        result = import_matrix_csv_input(
            period,
            args.matrix_dir / matrix_name,
            args.alpha_dir / alpha_name,
        )
        result.save(args.output_dir / f"{period}.npz")
        LOGGER.info("Imported %s (%d AGEBs).", period, result.n_patches)
    return 0


def _parameters_from_args(args: argparse.Namespace) -> EpidemicParameters:
    return EpidemicParameters(
        beta=args.beta,
        gamma=args.gamma,
        kappa=args.kappa,
        mu=args.mu,
        psi=args.psi,
        tau=args.tau,
    )


def _command_figure8(args: argparse.Namespace) -> int:
    comparisons = reproduce_figure8(
        args.input_dir,
        args.output_dir,
        parameters=_parameters_from_args(args),
        seed_agebs=args.seed_agebs,
        time_end=args.time_end,
        time_points=args.time_points,
        evaluation_day=args.evaluation_day,
    )
    for comparison in comparisons:
        LOGGER.info(
            "%s: %d common AGEBs; %.2f%% negative at day %.2f.",
            comparison.label,
            len(comparison.ageb_ids),
            100.0 * comparison.negative_fraction,
            comparison.evaluation_day,
        )
    return 0


def _command_demo(args: argparse.Namespace) -> int:
    input_dir = args.output_dir / "model_inputs"
    seed_agebs = write_synthetic_figure8_inputs(input_dir, n_patches=args.n_patches, seed=args.seed)
    reproduce_figure8(
        input_dir,
        args.output_dir,
        seed_agebs=seed_agebs,
        time_end=args.time_end,
        time_points=args.time_points,
        evaluation_day=args.evaluation_day,
    )
    LOGGER.info("Synthetic demonstration written to %s.", args.output_dir)
    return 0


def _add_repair_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repair-swapped-coordinates",
        dest="repair_swapped",
        action="store_true",
        help=(
            "repair the known latitude/longitude header inversion after validating the source file"
        ),
    )


def _add_epidemic_parameters(parser: argparse.ArgumentParser) -> None:
    defaults = EpidemicParameters()
    parser.add_argument("--beta", type=float, default=defaults.beta)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--kappa", type=float, default=defaults.kappa)
    parser.add_argument("--mu", type=float, default=defaults.mu)
    parser.add_argument("--psi", type=float, default=defaults.psi)
    parser.add_argument("--tau", type=float, default=defaults.tau)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="time-residence",
        description="Residence-occupation matrix estimation and epidemic simulation.",
    )
    parser.add_argument("--verbose", action="store_true", help="enable detailed logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assign_parser = subparsers.add_parser("assign-agebs", help="spatially assign pings to AGEBs")
    assign_parser.add_argument("--pings", type=Path, required=True)
    assign_parser.add_argument("--agebs", type=Path, required=True)
    assign_parser.add_argument("--output", type=Path, required=True)
    assign_parser.add_argument("--ageb-code-column", default="CVE_AGEB")
    _add_separator(assign_parser)
    _add_repair_flag(assign_parser)
    assign_parser.set_defaults(handler=_command_assign_agebs)

    metadata_parser = subparsers.add_parser(
        "prepare-agebs",
        help="create the stable AGEB index and population metadata table",
    )
    metadata_parser.add_argument("--agebs", type=Path, required=True)
    metadata_parser.add_argument("--output", type=Path, required=True)
    metadata_parser.add_argument("--ageb-code-column", default="CVE_AGEB")
    metadata_parser.add_argument("--population-column", default="POBTOT")
    _add_separator(metadata_parser)
    metadata_parser.set_defaults(handler=_command_prepare_agebs)

    period_parser = subparsers.add_parser("select-period", help="apply one Table 2 time window")
    period_parser.add_argument("--pings", type=Path, required=True)
    period_parser.add_argument("--period", choices=sorted(STUDY_PERIODS), required=True)
    period_parser.add_argument("--eligible-ids", type=Path)
    period_parser.add_argument("--id-column", default="id_adv")
    period_parser.add_argument("--output", type=Path, required=True)
    _add_separator(period_parser)
    _add_repair_flag(period_parser)
    period_parser.set_defaults(handler=_command_select_period)

    residence_parser = subparsers.add_parser(
        "assign-residences", help="infer one residence AGEB per device"
    )
    residence_parser.add_argument("--pings", type=Path, required=True)
    residence_parser.add_argument("--ageb-metadata", type=Path, required=True)
    residence_parser.add_argument("--output", type=Path, required=True)
    residence_parser.add_argument("--population-column", default="POBTOT")
    residence_parser.add_argument("--seed", type=int, default=2025)
    _add_separator(residence_parser)
    _add_repair_flag(residence_parser)
    residence_parser.set_defaults(handler=_command_assign_residences)

    rom_parser = subparsers.add_parser(
        "estimate-individual-roms", help="estimate Brownian-bridge occupation vectors"
    )
    rom_parser.add_argument("--pings", type=Path, required=True)
    rom_parser.add_argument("--agebs", type=Path, required=True)
    rom_parser.add_argument("--output", type=Path, required=True)
    rom_parser.add_argument("--parameter-output", type=Path, required=True)
    rom_parser.add_argument("--failure-output", type=Path, default=Path("outputs/failures.json"))
    rom_parser.add_argument("--samples", type=int, default=1000)
    rom_parser.add_argument("--seed", type=int, default=2025)
    location_group = rom_parser.add_mutually_exclusive_group()
    location_group.add_argument(
        "--location-error",
        type=float,
        default=DEFAULT_LOCATION_ERROR_METERS,
        help="fixed location-error standard deviation in metres (default: 28.85)",
    )
    location_group.add_argument(
        "--estimate-location-error",
        action="store_true",
        help="jointly estimate sigma and location error from each trajectory",
    )
    rom_parser.add_argument(
        "--projected-crs",
        default=ARTICLE_PROJECTED_CRS,
        help="projected CRS used for metric calculations (default: EPSG:32612, UTM zone 12N)",
    )
    rom_parser.add_argument("--ageb-code-column", default="CVE_AGEB")
    rom_parser.add_argument("--fail-fast", action="store_true")
    _add_separator(rom_parser)
    _add_repair_flag(rom_parser)
    rom_parser.set_defaults(handler=_command_estimate_individual_roms)

    build_parser_ = subparsers.add_parser(
        "build-model-input", help="align alpha, conditional ROM, population, and AGEB ids"
    )
    build_parser_.add_argument("--period", choices=sorted(STUDY_PERIODS), required=True)
    build_parser_.add_argument("--individual-roms", type=Path, required=True)
    build_parser_.add_argument("--pings", type=Path, required=True)
    build_parser_.add_argument("--residences", type=Path, required=True)
    build_parser_.add_argument("--ageb-metadata", type=Path, required=True)
    build_parser_.add_argument("--population-column", default="POBTOT")
    build_parser_.add_argument("--output", type=Path, required=True)
    _add_separator(build_parser_)
    _add_repair_flag(build_parser_)
    build_parser_.set_defaults(handler=_command_build_model_input)

    import_parser = subparsers.add_parser(
        "import-article-inputs",
        help="convert the Figure 8 matrix/CSV inputs to aligned NPZ files",
    )
    import_parser.add_argument("--matrix-dir", type=Path, required=True)
    import_parser.add_argument("--alpha-dir", type=Path, required=True)
    import_parser.add_argument("--output-dir", type=Path, required=True)
    import_parser.set_defaults(handler=_command_import_article_inputs)

    figure_parser = subparsers.add_parser("figure8", help="reproduce the six Figure 8 panels")
    figure_parser.add_argument("--input-dir", type=Path, required=True)
    figure_parser.add_argument("--output-dir", type=Path, required=True)
    figure_parser.add_argument("--seed-agebs", nargs="+", default=list(DEFAULT_SEED_AGEBS))
    figure_parser.add_argument("--time-end", type=float, default=200.0)
    figure_parser.add_argument("--time-points", type=int, default=100)
    figure_parser.add_argument("--evaluation-day", type=float, default=30.0)
    _add_epidemic_parameters(figure_parser)
    figure_parser.set_defaults(handler=_command_figure8)

    demo_parser = subparsers.add_parser("demo", help="run Figure 8 on deterministic synthetic data")
    demo_parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    demo_parser.add_argument("--n-patches", type=int, default=12)
    demo_parser.add_argument("--seed", type=int, default=2025)
    demo_parser.add_argument("--time-end", type=float, default=60.0)
    demo_parser.add_argument("--time-points", type=int, default=61)
    demo_parser.add_argument("--evaluation-day", type=float, default=30.0)
    demo_parser.set_defaults(handler=_command_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 2
