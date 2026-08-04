import itertools
import json

import numpy as np
import pandas as pd
from project_paths import DATA_ROOT, prepare_output
from tqdm import tqdm

tqdm.pandas()


# rng = np.random.default_rng()
def choice_func(row, population):
    a1, a2 = set(row["ageb_crit1"]), set(row["ageb_crit2"])
    common = list(a1.intersection(a2))
    if len(common) == 0:
        return np.random.choice(list(a2))
    elif len(common) == 1:
        return common.pop()
    else:
        prob = np.array([population.get(str(ageb), 0) for ageb in common])
        if prob.sum() == 0:
            return -1
        prob = prob / prob.sum()
        return np.random.choice(common, p=prob)


CREATE_DATA = True
for period, part in tqdm(itertools.product(["First", "Second", "Third"], ["First", "Second"])):
    if CREATE_DATA:
        for crit in [1, 2]:
            df = pd.read_csv(
                DATA_ROOT
                / "final-time-periods"
                / f"criterion_{crit}"
                / f"{period}Period_{part}Part_final.csv",
                sep=";",
            )
            df_grouped = df.groupby("id")
            temp = {}
            for idx, group in tqdm(df_grouped, leave=False):
                temp[idx] = group["polygon"].mode().tolist()

            json.dump(
                temp,
                open(
                    prepare_output(
                        DATA_ROOT
                        / "final-residence-agebs"
                        / f"criterion_{crit}"
                        / f"{period}Period_{part}Part_agebs.json"
                    ),
                    "w",
                ),
                indent=4,
            )

    df_crit1 = pd.DataFrame(
        pd.read_json(
            open(
                DATA_ROOT
                / "final-residence-agebs"
                / "criterion_1"
                / f"{period}Period_{part}Part_agebs.json",
            ),
            orient="index",
            typ="series",
        ),
        columns=["ageb"],
    )
    df_crit2 = pd.DataFrame(
        pd.read_json(
            open(
                DATA_ROOT
                / "final-residence-agebs"
                / "criterion_2"
                / f"{period}Period_{part}Part_agebs.json",
            ),
            orient="index",
            typ="series",
        ),
        columns=["ageb"],
    )

    population_dict = json.load(open(DATA_ROOT / "ageb-population-mapping.json"))
    joined = df_crit2.join(df_crit1, lsuffix="_crit2", rsuffix="_crit1")
    joined["loose"] = joined.progress_apply(choice_func, axis=1, args=(population_dict,))

    joined.to_csv(
        prepare_output(
            DATA_ROOT / "final-residence-agebs" / "combined" / f"{period}Period_{part}Part_comb.csv"
        ),
        sep=";",
    )
