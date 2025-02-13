import os
import json
import random
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm

tqdm.pandas()
from collections import Counter

# rng = np.random.default_rng()
def choice_func(row, population):
    a1, a2 = set(row["ageb_crit1"]), set(row["ageb_crit2"])
    common = list(a1.intersection(a2))
    if len(common) == 0:
        return np.random.choice(list(a2))
    elif len(common) == 1:
        return common.pop()
    else:
        # import pdb;pdb.set_trace()
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
                f"/workspace/CHAHAK/bbmm/final-time-periods/criterion_{crit}/{period}Period_{part}Part_final.csv",
                sep=";",
            )
            df_grouped = df.groupby("id")
            temp = {}
            for idx, group in tqdm(df_grouped, leave=False):
                # import pdb;pdb.set_trace()
                temp[idx] = group["polygon"].mode().tolist()

            json.dump(
                temp,
                open(
                    f"/workspace/CHAHAK/bbmm/final-residence-agebs/criterion_{crit}/{period}Period_{part}Part_agebs.json",
                    "w",
                ),
                indent=4,
            )

    # import pdb; pdb.set_trace()
    df_crit1 = pd.DataFrame(
        pd.read_json(
            open(
                f"/workspace/CHAHAK/bbmm/final-residence-agebs/criterion_1/{period}Period_{part}Part_agebs.json",
                "r",
            ),
            orient="index",
            typ="series",
        ),
        columns=["ageb"],
    )
    df_crit2 = pd.DataFrame(
        pd.read_json(
            open(
                f"/workspace/CHAHAK/bbmm/final-residence-agebs/criterion_2/{period}Period_{part}Part_agebs.json",
                "r",
            ),
            orient="index",
            typ="series",
        ),
        columns=["ageb"],
    )

    population_dict = json.load(open("/workspace/CHAHAK/bbmm/ageb-population-mapping.json", "r"))
    joined = df_crit2.join(df_crit1, lsuffix="_crit2", rsuffix="_crit1")
    joined["loose"] = joined.progress_apply(choice_func, axis=1, args=(population_dict,))

    joined.to_csv(
        f"/workspace/CHAHAK/bbmm/final-residence-agebs/combined/{period}Period_{part}Part_comb.csv",
        sep=";",
    )
