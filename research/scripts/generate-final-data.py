import itertools

import pandas as pd
from project_paths import DATA_ROOT, prepare_output
from tqdm import tqdm

pd.set_option("display.max_columns", 999)
df_m1 = pd.read_csv(DATA_ROOT / "ageb_M1_sorted.csv.zip", compression="zip")
print("Loaded M1")
df_m2 = pd.read_csv(DATA_ROOT / "ageb_M2_sorted.csv.zip", compression="zip")
print("Loaded M2")
df_m3 = pd.read_csv(DATA_ROOT / "ageb_M3_sorted.csv.zip", compression="zip")
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


for period, part in tqdm(itertools.product(["First", "Second", "Third"], ["First", "Second"])):
    df_id_pp = pd.read_csv(DATA_ROOT / f"{period}Period_{part}Part.csv")

    df_pp = df_full.loc[df_full["id"].isin(df_id_pp["id_adv"])]
    tqdm.write(str(df_pp.shape))

    start_date, end_date = timeframes[f"{period.lower()}_{part.lower()}"]
    local_time = df_pp["timestamp"].dt.tz_convert("America/Hermosillo")
    local_date = local_time.dt.date
    df_pp = df_pp.loc[
        (local_date >= pd.Timestamp(start_date).date())
        & (local_date <= pd.Timestamp(end_date).date())
    ]

    df_pp.to_csv(
        prepare_output(
            DATA_ROOT
            / "final-time-periods"
            / "criterion_1"
            / f"{period}Period_{part}Part_final.csv"
        ),
        sep=";",
        index=False,
    )
    timestamps = df_pp["timestamp"].dt.tz_convert("America/Hermosillo")
    df_pp = df_pp.loc[(timestamps.dt.hour < 6) | (timestamps.dt.hour >= 22)]
    df_pp.to_csv(
        prepare_output(
            DATA_ROOT
            / "final-time-periods"
            / "criterion_2"
            / f"{period}Period_{part}Part_final.csv"
        ),
        sep=";",
        index=False,
    )

    del df_pp
