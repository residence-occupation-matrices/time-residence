import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from tqdm import tqdm

tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument("--id", action="store_true")
parser.add_argument("--pin", action="store_true")
parser.add_argument(
    "--period",
    type=str,
    choices=["First", "Second", "Third"],
    help="Time period to run",
    required=True,
)
parser.add_argument(
    "--part",
    type=str,
    choices=["First", "Second"],
    help="Part in time period to run",
    required=True,
)

args = parser.parse_args()

PERIOD = args.period
PART = args.part

hermosillo = gpd.read_file("/workspace/CHAHAK/bbmm/geometry/26a.shp")
sorted_agebs_mapping = pd.read_csv("/workspace/CHAHAK/bbmm/sorted-ageb-mapping.csv")
# Plot number of IDs for each AGEB
if args.id:
    df_residence = pd.read_csv(
        f"/workspace/CHAHAK/bbmm/final-residence-agebs/combined/{PERIOD}Period_{PART}Part_comb.csv",
        sep=";",
    )

    count_df = df_residence["loose"].value_counts().to_dict()
    hermosillo["id_count"] = hermosillo["CVE_AGEB"].progress_apply(
        lambda x: count_df.get(
            sorted_agebs_mapping.loc[sorted_agebs_mapping["CVE_AGEB"] == x, "ageb_id"].values[0], 0
        )
    )
    print(hermosillo.loc[hermosillo["id_count"] == 0].shape)
    fig, ax = plt.subplots(2, 1, figsize=(8, 16))
    hermosillo.plot(ax=ax[0], column="id_count", legend=True)
    ax[1].bar(range(hermosillo.shape[0]), hermosillo["id_count"])
    fig.savefig(
        f"/oden/cmehta/Documents/residence-time/figures/hermosillo_ageb_id_count_{PERIOD}_{PART}.png",
        dpi=350,
    )


# Plot number of pins for each AGEB
if args.pin:
    hermosillo["pin_count"] = 0
    for i in tqdm([1, 2, 3]):
        df_m = pd.read_csv(f"/workspace/CHAHAK/bbmm/ageb_M{i}_sorted.csv.zip", compression="zip")
        count_df = df_m["polygon"].value_counts().to_dict()
        hermosillo["pin_count"] = hermosillo["CVE_AGEB"].apply(
            lambda x: count_df.get(
                sorted_agebs_mapping.loc[sorted_agebs_mapping["CVE_AGEB"] == x, "ageb_id"].values[
                    0
                ],
                0,
            )
        )

        del df_m
        del count_df

    fig, ax = plt.subplots(figsize=(10, 10))
    hermosillo.plot(ax=ax, column="pin_count", legend=True)
    fig.savefig(
        "/oden/cmehta/Documents/residence-time/figures/hermosillo_ageb_pin_count.png", dpi=350
    )
