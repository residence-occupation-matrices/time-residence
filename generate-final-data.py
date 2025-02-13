import itertools
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 999)
from tqdm import tqdm

df_m1 = pd.read_csv("/workspace/CHAHAK/bbmm/ageb_M1_sorted.csv.zip", compression="zip")
print("Loaded M1")
df_m2 = pd.read_csv("/workspace/CHAHAK/bbmm/ageb_M2_sorted.csv.zip", compression="zip")
print("Loaded M2")
df_m3 = pd.read_csv("/workspace/CHAHAK/bbmm/ageb_M3_sorted.csv.zip", compression="zip")
print("Loaded M3")
df_full = pd.concat([df_m1, df_m2, df_m3])
del df_m1
del df_m2
del df_m3
df_full["timestamp"] = df_full["timestamp"].astype("datetime64[ns, UTC]")
print(df_full.shape)

timeframes = {
    "first_first": ["2020-09-21", "2020-10-04"],
    "first_second": ["2020-10-26", "2020-11-08"],
    "second_first": ["2020-09-21", "2020-10-04"],
    "second_second": ["2020-11-02", "2020-11-15"],
    "third_first": ["2020-09-21", "2020-10-11"],
    "third_second": ["2020-10-12", "2020-11-01"],
}


# import pdb; pdb.set_trace()
for period, part in tqdm(itertools.product(["First", "Second", "Third"], ["First", "Second"])):
    # tqdm.set_description(desc=f'{period}-{part}')
    df_id_pp = pd.read_csv(f"/workspace/CHAHAK/bbmm/{period}Period_{part}Part.csv")

    df_pp = df_full.loc[df_full["id"].isin(df_id_pp["id_adv"])]
    tqdm.write(str(df_pp.shape))

    t0, t1 = timeframes[f"{period.lower()}_{part.lower()}"]
    t0, t1 = pd.Timestamp(t0, tz="UTC"), pd.Timestamp(t1, tz="UTC")

    df_pp = df_pp.loc[(df_pp["timestamp"] > t0) & (df_pp["timestamp"] < t1)]

    df_pp.to_csv(
        f"/workspace/CHAHAK/bbmm/final-time-periods/criterion_1/{period}Period_{part}Part_final.csv",
        sep=";",
        index=False,
    )
    timestamps = df_pp["timestamp"].apply(pd.Timestamp.astimezone, args=("MST",))
    df_pp = df_pp.loc[(timestamps.dt.hour <= 6) | (timestamps.dt.hour >= 22)]
    df_pp.to_csv(
        f"/workspace/CHAHAK/bbmm/final-time-periods/criterion_2/{period}Period_{part}Part_final.csv",
        sep=";",
        index=False,
    )

    del df_pp
