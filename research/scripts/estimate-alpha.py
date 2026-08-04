import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from project_paths import DATA_ROOT, OUTPUT_ROOT, prepare_output
from tqdm import tqdm

tqdm.pandas()

parser = argparse.ArgumentParser(description="Alpha estimation")
parser.add_argument(
    "--period", type=str, choices=["First", "Second", "Third"], help="Time period to run"
)
parser.add_argument(
    "--part", type=str, choices=["First", "Second"], help="Part in time period to run"
)

args = parser.parse_args()

PERIOD = args.period
PART = args.part

matrices_dict = json.load(
    open(
        DATA_ROOT / "final_residence_matrices" / f"final_residence_matrices_{PERIOD}_{PART}.json",
    )
)

df_ids = pd.read_csv(
    DATA_ROOT / "final-residence-agebs" / "combined" / f"{PERIOD}Period_{PART}Part_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)


for k, v in tqdm(matrices_dict.items()):
    v = np.array(v)
    matrices_dict[k] = v / v.sum() if v.sum() != 0 else v


alpha_values = []
# threshold_values = np.arange(0.001, 1, 0.001)
threshold_values = [0.05]
PLOT = True if len(threshold_values) > 1 else False
for threshold in tqdm(threshold_values):
    # for threshold in tqdm([0.1]):
    thresholded_dict = {}
    # threshold = 0.01

    for k, v in matrices_dict.items():
        thresholded_dict[k] = np.array(v) > threshold

    temp = []
    # for ageb, group in tqdm(df_ids.groupby(by="loose"), leave=False):
    for ageb in range(582):
        group = df_ids.loc[df_ids["loose"] == ageb]
        if len(group) == 0:
            continue
        count = 0
        for idx in tqdm(group["id"].unique(), leave=False):
            count += 1 if (thresholded_dict[idx] > 0).sum() > 1 else 0
        temp.append(count / group.shape[0] if group.shape[0] > 0 else 0)

    # print(f"Average alpha value: {np.array(temp).mean()}")
    if not PLOT:
        print(len(temp))
        json.dump(
            {"alpha_values": temp},
            open(
                prepare_output(
                    DATA_ROOT / "alpha_values" / f"albert_alpha_values_{PERIOD}_{PART}.json"
                ),
                "w",
            ),
            indent=4,
        )
    alpha_values.append(np.array(temp).mean())

if PLOT:
    fig, ax = plt.subplots()
    ax.plot(np.arange(0.001, 1, 0.001), alpha_values, "k-o", markersize=1.2, linewidth=1)
    ax.grid()
    ax.set_xlabel("threshold")
    ax.set_ylabel("alpha")
    fig.savefig(
        prepare_output(OUTPUT_ROOT / f"alpha_variations_sum_normalized_{PERIOD}_{PART}.png"),
        dpi=350,
    )
