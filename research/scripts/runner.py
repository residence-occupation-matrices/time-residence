import argparse
import json
import logging
import os

import pandas as pd
from pandarallel import pandarallel
from project_paths import DATA_ROOT, OUTPUT_ROOT, prepare_output
from residence import calculate_residence_matrix, calculate_sigma
from tqdm import tqdm

tqdm.pandas()


parser = argparse.ArgumentParser(
    description="Estimate device motion parameters or occupation vectors."
)
parser.add_argument(
    "--period",
    type=str,
    choices=["First", "Second", "Third"],
    required=True,
    help="study period to process",
)
parser.add_argument(
    "--part",
    type=str,
    choices=["First", "Second"],
    required=True,
    help="part of the study period to process",
)
parser.add_argument(
    "--exp",
    type=str,
    choices=["sigma", "resmat"],
    required=True,
    help="quantity to estimate",
)
parser.add_argument(
    "--workers",
    type=int,
    default=os.cpu_count() or 1,
    help="parallel worker count (default: available CPUs)",
)

args = parser.parse_args()
if args.workers < 1:
    parser.error("--workers must be at least 1")
pandarallel.initialize(nb_workers=args.workers, progress_bar=True)

PERIOD = args.period
PART = args.part

logging.basicConfig(
    filename=prepare_output(OUTPUT_ROOT / f"runner_{PERIOD}_{PART}.log"),
    filemode="w",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

logger.debug("Loading GPS data")
df_ids = pd.read_csv(
    DATA_ROOT / "final-residence-agebs" / "combined" / f"{PERIOD}Period_{PART}Part_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)
df_ids["loose"] = df_ids["loose"].astype(int)
df_full = pd.read_csv(
    DATA_ROOT / "final-time-periods" / "criterion_1" / f"{PERIOD}Period_{PART}Part_final.csv",
    sep=";",
    header=0,
    names=["id_adv", "timestamp", "lat", "lon", "polygon"],
)
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], utc=True)
ids = set(df_full["id_adv"].unique())

logger.debug("All data loaded")


def get_data(id_adv):
    return df_full.loc[df_full["id_adv"] == id_adv]


def calc_res_mat(row):
    df_data = get_data(row["id"])
    return calculate_residence_matrix(df_data, row["loose"])


def calc_sigma(row):
    df_data = get_data(row["id"])
    return calculate_sigma(df_data)


def runner_res_mat():
    residence_matrices = {}
    df_ids["matrix"] = df_ids.parallel_apply(calc_res_mat, axis=1)

    for _i, row in tqdm(df_ids.iterrows()):
        residence_matrices[row["id"]] = row["matrix"]
    json.dump(
        residence_matrices,
        open(
            prepare_output(
                DATA_ROOT / "new_sigma_m" / f"final_residence_matrices_{PERIOD}_{PART}.json"
            ),
            "w",
        ),
        indent=4,
    )
    return


def runner_sigma():
    sigma_values = {}
    df_ids["sigma"] = df_ids.parallel_apply(calc_sigma, axis=1)

    for _i, row in tqdm(df_ids.iterrows()):
        sigma_values[row["id"]] = row["sigma"]
    json.dump(
        sigma_values,
        open(
            prepare_output(DATA_ROOT / "new_sigma_m" / f"sigma_m_{PERIOD}_{PART}.json"),
            "w",
        ),
        indent=4,
    )
    return


if __name__ == "__main__":
    if args.exp == "sigma":
        runner_sigma()
    elif args.exp == "resmat":
        runner_res_mat()
