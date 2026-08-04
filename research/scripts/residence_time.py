import geopandas as gpd
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from project_paths import DATA_ROOT
from scipy.optimize import minimize_scalar

pandarallel.initialize()

data_ageb = gpd.read_file(DATA_ROOT / "geometry" / "26a.shp")
# print(data_ageb.head())

# data_ageb.crs

data_ageb.to_crs("EPSG:3857", inplace=True)
data_ageb.plot(figsize=(6, 6))

# data_ageb.crs

df_ind = pd.read_csv(DATA_ROOT / "private" / "example_device_pings.csv").drop_duplicates(
    subset="timestamp"
)
print(df_ind.shape)
# print(df_ind.head(3))

df_ind.drop(columns="id_adv", inplace=True)
# print(df_ind.head(3))

df_ind["timestamp"] = df_ind["timestamp"].astype("datetime64[ns, UTC]")
df_ind["timestamp"].head(3)

gdf_ind = gpd.GeoDataFrame(
    df_ind, geometry=gpd.points_from_xy(df_ind["lon"], df_ind["lat"], crs="EPSG:4326")
).sort_values(by="timestamp")
# print(gdf_ind.head(3))

gdf_ind.to_crs("EPSG:3857", inplace=True)
# print(gdf_ind.head(3))


def calculate_polygon(point, data_ageb):
    for i, pol in enumerate(data_ageb["geometry"]):
        if pol.contains(point):
            return i
    return -1


gdf_ind["polygon"] = gdf_ind["geometry"].apply(calculate_polygon, args=(data_ageb,))
# print(gdf_ind.head(3))
# data_ageb["geometry"].contains(gdf_ind["geometry"].iloc[0]).argmax()

count_df = gdf_ind["polygon"].value_counts()
residence_polygon = count_df.index[0] if count_df.index[0] != -1 else count_df.index[1]
print(residence_polygon)

gdf_timestamp_mst = gdf_ind["timestamp"].apply(lambda t: pd.Timestamp.astimezone(t, "MST"))
count_df = gdf_ind.loc[(gdf_timestamp_mst.dt.hour >= 22) | (gdf_timestamp_mst.dt.hour < 6)][
    "polygon"
].value_counts()
residence_polygon = count_df.index[0] if count_df.index[0] != -1 else count_df.index[1]
print(residence_polygon)


def L_mod(gdf_ind, sigma_m, delta_z):
    result = 0
    for i in range(0, len(gdf_ind) - 2, 2):
        start_point = gdf_ind.iloc[i]
        mid_point = gdf_ind.iloc[i + 1]
        end_point = gdf_ind.iloc[i + 2]
        T_i = (end_point["timestamp"] - start_point["timestamp"]).total_seconds()
        alpha = (mid_point["timestamp"] - start_point["timestamp"]).total_seconds() / T_i
        mu_tx = start_point["geometry"].x + alpha * (
            end_point["geometry"].x - start_point["geometry"].x
        )
        mu_ty = start_point["geometry"].y + alpha * (
            end_point["geometry"].y - start_point["geometry"].y
        )
        sigma_t = np.sqrt(
            T_i * alpha * (1 - alpha) * sigma_m**2 + ((1 - alpha) ** 2 + alpha**2) * delta_z**2
        )
        result += 2 * np.log(sigma_t) + (
            ((mid_point["geometry"].x - mu_tx) ** 2 + (mid_point["geometry"].y - mu_ty) ** 2)
            / (2 * sigma_t**2)
        )

    return result


def calculate_sigma_m(gdf_ind, delta_z):
    res = minimize_scalar(lambda sigma_m: L_mod(gdf_ind, sigma_m, delta_z))
    sigma_m = res.x
    return sigma_m


def norm_pdf(x, mu, variance):
    mu, variance = mu.reshape(-1, 1), variance.reshape(-1, 1)
    x = x.reshape(-1, 1)
    # variance = sigma**2
    numerator = x - mu
    denominator = 2 * variance
    pdf = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-(numerator**2) / denominator)
    return pdf


rng1 = np.random.default_rng(seed=10)
rng2 = np.random.default_rng(seed=10)


def h_z(a, b, T_i, x, y, sigma_m2, delta_a, delta_b, indicator, rng, n_time_samples=1000):
    mc_sum = np.zeros(x.shape)
    t = rng.uniform(0, T_i, size=n_time_samples)
    alpha = t / T_i
    ax, ay, bx, by = a.x, a.y, b.x, b.y
    mu_x = ax + alpha * (bx - ax)
    mu_y = ay + alpha * (by - ay)
    variance = (
        t * (1 - alpha) * sigma_m2 + (1 - alpha) ** 2 * (delta_a**2) + (alpha**2) * (delta_b**2)
    )
    pdf_x = norm_pdf(x, mu_x, variance)
    pdf_y = norm_pdf(y, mu_y, variance)
    mc_sum += indicator * pdf_x * pdf_y

    return mc_sum


def prob_region(data, cve_ageb, gdf_ind, sigma_m, delta_z, M, rng):
    N = len(gdf_ind)
    polygon = data.loc[data["CVE_AGEB"] == cve_ageb, "geometry"]
    limits = polygon.bounds.iloc[0]
    rx = rng.uniform(limits["minx"], limits["maxx"], M).reshape(-1, 1)
    ry = rng.uniform(limits["miny"], limits["maxy"], M).reshape(-1, 1)
    points = gpd.GeoSeries(gpd.points_from_xy(rx, ry, crs="EPSG:3857"))
    indicator = points.within(polygon.iloc[0]).to_numpy().reshape(-1, 1)
    T_total = (gdf_ind["timestamp"].max() - gdf_ind["timestamp"].min()).total_seconds()
    tot = 0
    for i in range(N - 1):
        start_point = gdf_ind.iloc[i]
        end_point = gdf_ind.iloc[i + 1]
        T_i = (end_point["timestamp"] - start_point["timestamp"]).total_seconds()
        a, b = start_point["geometry"], end_point["geometry"]
        prob = h_z(
            a, b, T_i, rx, ry, sigma_m**2, delta_z, delta_z, indicator, rng, n_time_samples=M
        )
        accum = prob.sum()
        tot += (
            (limits["maxx"] - limits["minx"])
            * (limits["maxy"] - limits["miny"])
            * accum
            * (T_i / M)
        )

    return tot / T_total


gdf_ind["next_point"] = gdf_ind["geometry"].shift(-1)
gdf_ind["next_time"] = gdf_ind["timestamp"].shift(-1)


def prob_region_new(data, cve_ageb, gdf_ind, sigma_m, delta_z, M, rng):
    polygon = data.loc[data["CVE_AGEB"] == cve_ageb, "geometry"]
    limits = polygon.bounds.iloc[0]
    rx = rng.uniform(limits["minx"], limits["maxx"], M).reshape(-1, 1)
    ry = rng.uniform(limits["miny"], limits["maxy"], M).reshape(-1, 1)
    points = gpd.GeoSeries(gpd.points_from_xy(rx, ry, crs="EPSG:3857"))
    indicator = points.within(polygon.iloc[0]).to_numpy().reshape(-1, 1)
    T_total = (gdf_ind["timestamp"].max() - gdf_ind["timestamp"].min()).total_seconds()
    gdf_ind["T_i"] = (gdf_ind["next_time"] - gdf_ind["timestamp"]).dt.total_seconds()
    gdf_ind["hz"] = gdf_ind.apply(
        lambda row: (
            h_z(
                row["geometry"],
                row["next_point"],
                row["T_i"],
                rx,
                ry,
                sigma_m**2,
                delta_z,
                delta_z,
                indicator,
                rng,
                n_time_samples=M,
            ).sum()
            * row["T_i"]
            if pd.notna(row["T_i"])
            else 0
        ),
        axis=1,
    )
    res = (
        (limits["maxx"] - limits["minx"])
        * (limits["maxy"] - limits["miny"])
        * gdf_ind["hz"].sum()
        / M
    )
    res = res / T_total
    # res = gdf_ind["accum"].sum()/T_total
    del gdf_ind["T_i"]
    del gdf_ind["hz"]
    # del gdf_ind["accum"]
    return res


delta_z = 28.85
print("Calculating sigma_m")
sigma_m = calculate_sigma_m(gdf_ind, delta_z)
# sigma_m = 73.78817684857587
print(sigma_m)
d = prob_region_new(
    data_ageb,
    data_ageb["CVE_AGEB"].unique()[residence_polygon],
    gdf_ind,
    sigma_m,
    delta_z,
    1000,
    rng2,
)
print(d)
d = prob_region(
    data_ageb,
    data_ageb["CVE_AGEB"].unique()[residence_polygon],
    gdf_ind,
    sigma_m,
    delta_z,
    1000,
    rng1,
)
print(d)

# prob_dist = np.zeros(len(data_ageb))
# for i, ageb in tqdm(enumerate(data_ageb['CVE_AGEB'].unique())):
#     prob_dist[i] = prob_region_new(data_ageb, ageb, gdf_ind, sigma_m, delta_z, 1000)

# fig, ax = plt.subplots()
# ax.plot(range(len(prob_dist)), prob_dist, linewidth=1)
# plt.show()
