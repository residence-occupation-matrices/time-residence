import argparse
import json
import logging

import pandas as pd
from pandarallel import pandarallel
from project_paths import DATA_ROOT, prepare_output
from residence import calculate_residence_matrix, calculate_sigma
from tqdm import tqdm

tqdm.pandas()
pandarallel.initialize(nb_workers=48, progress_bar=True)

logging.basicConfig(filename="run.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Residence-time runner file.")
parser.add_argument(
    "--period", type=str, choices=["First", "Second", "Third"], help="Time period to run."
)
parser.add_argument(
    "--part", type=str, choices=["First", "Second"], help="Part in time period to run."
)
parser.add_argument("--exp", type=str, choices=["sigma", "resmat"], help="Experiment to run")

args = parser.parse_args()

PERIOD = args.period
PART = args.part

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
df_full["timestamp"] = df_full["timestamp"].astype("datetime64[ns, UTC]")
ids = set(df_full["id_adv"].unique())
# df_m2["timestamp"] = df_m2["timestamp"].astype("datetime64[ns, UTC]")
# df_m3["timestamp"] = df_m3["timestamp"].astype("datetime64[ns, UTC]")
# m1_ids = set(df_m1["id_adv"].unique())
# m2_ids = set(df_m2["id_adv"].unique())
# m3_ids = set(df_m3["id_adv"].unique())

logger.debug("All data loaded")


def get_data(id_adv):
    # if id_adv in m1_ids:
    #     return df_m1.loc[df_m1["id_adv"]==id_adv].drop_duplicates(subset="timestamp")
    # elif id_adv in m2_ids:
    #     return df_m2.loc[df_m2["id_adv"]==id_adv].drop_duplicates(subset="timestamp")
    # elif id_adv in m3_ids:
    #     return df_m3.loc[df_m3["id_adv"]==id_adv].drop_duplicates(subset="timestamp")
    # else:
    #     raise ValueError(f"id not found in any file: {id_adv}")
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
