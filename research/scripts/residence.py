import hashlib
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.linalg
from project_paths import DATA_ROOT
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import dia_matrix
from tqdm import tqdm

logger = logging.getLogger(__name__)

delta_z = 28.85
gdf_ageb = gpd.read_file(DATA_ROOT / "geometry" / "26a.shp")
gdf_ageb.to_crs("EPSG:3857", inplace=True)
gdf_ageb.sort_values(by="CVE_AGEB", inplace=True)

rng = np.random.default_rng(seed=10)


def calculate_polygon(point, gdf_ageb):
    for i, pol in enumerate(gdf_ageb["geometry"]):
        if pol.contains(point):
            return i
    return -1


def prepare_geometry_data(df):
    df.drop(columns="id_adv", inplace=True)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"),
    ).sort_values(by="timestamp")
    gdf.to_crs("EPSG:3857", inplace=True)
    gdf["polygon"] = gdf["geometry"].apply(calculate_polygon, args=(gdf_ageb,))
    gdf["next_point"] = gdf["geometry"].shift(-1)
    gdf["next_time"] = gdf["timestamp"].shift(-1)

    return gdf


def get_sparse_sigma(time: np.ndarray, parameter: np.ndarray):
    n = len(time)
    tau = time.diff().fillna(pd.Timedelta(seconds=0)).apply(pd.Timedelta.total_seconds)
    tau.drop(index=tau.index[0], inplace=True)
    d0 = tau * parameter[0] ** 2 + 2 * delta_z**2
    d1 = np.ones(n - 1) * -(delta_z**2)
    offsets = np.array([-1, 0, 1])
    data = np.array([d1, d0, d1])
    sigma_diag = dia_matrix((data, offsets), shape=(n - 1, n - 1)).toarray()
    return sigma_diag


def multivariate_density(x: np.ndarray, cov_mat):
    n = cov_mat.shape[0]
    c, lower = scipy.linalg.cho_factor(cov_mat)
    p1 = -(n / 2) * np.log(2 * np.pi) - np.log(np.diag(c)).sum()
    y = scipy.linalg.cho_solve((c, lower), x)
    p2 = -(np.dot(x, y)) / 2
    return p1 + p2


def negative_loglikelihood(parameters: np.ndarray, df: pd.DataFrame):
    zxi_s = df["geometry"].x.diff()
    zxi_s.drop(index=zxi_s.index[0], inplace=True)
    sigma_x = get_sparse_sigma(df["timestamp"], parameters)
    loglike_x = multivariate_density(zxi_s, sigma_x)

    zyi_s = df["geometry"].y.diff()
    zyi_s.drop(index=zyi_s.index[0], inplace=True)
    sigma_y = get_sparse_sigma(df["timestamp"], parameters)
    loglike_y = multivariate_density(zyi_s, sigma_y)

    likelihood = -(loglike_x + loglike_y)
    return likelihood


def calculate_parameters(df):
    tau = df["timestamp"].diff().fillna(pd.Timedelta(seconds=0)).apply(pd.Timedelta.total_seconds)
    tau.drop(index=tau.index[0], inplace=True)
    zxi_s = df["geometry"].x.diff()
    zyi_s = df["geometry"].y.diff()
    start_val = np.sqrt((zxi_s**2 + zyi_s**2).sum() / (2 * (2 + tau).sum()))
    result = minimize(negative_loglikelihood, [start_val], args=(df,))
    if result.success and np.isfinite(result.x[0]) and result.x[0] > 0:
        value = float(result.x[0])
    else:
        tqdm.write("Minimizer did not converge. Using start value.")
        value = start_val
    return value


def L_mod(gdf, sigma_m, delta_z):
    result = 0
    for i in range(0, len(gdf) - 2, 2):
        start = gdf.iloc[i]
        mid = gdf.iloc[i + 1]
        end = gdf.iloc[i + 2]
        T_i = (end["timestamp"] - start["timestamp"]).total_seconds()
        alpha = (mid["timestamp"] - start["timestamp"]).total_seconds() / T_i
        mu_tx = start["geometry"].x + alpha * (end["geometry"].x - start["geometry"].x)
        mu_ty = start["geometry"].y + alpha * (end["geometry"].y - start["geometry"].y)
        sigma_t = np.sqrt(
            T_i * alpha * (1 - alpha) * sigma_m**2 + ((1 - alpha) ** 2 + alpha**2) * delta_z**2
        )
        result += 2 * np.log(sigma_t) + (
            ((mid["geometry"].x - mu_tx) ** 2 + (mid["geometry"].y - mu_ty) ** 2) / (2 * sigma_t**2)
        )

    return result


def calculate_sigma_m(gdf, delta_z):
    result_old = minimize_scalar(lambda sigma_m: L_mod(gdf, sigma_m, delta_z))
    result = calculate_parameters(gdf)
    sigma_m = result
    return [float(result_old.x), sigma_m]


def prob_region(data, ageb, gdf, sigma_m, delta_z, M, rng):
    polygon = data.loc[data["CVE_AGEB"] == ageb, "geometry"]
    limits = polygon.bounds.iloc[0]
    rx = rng.uniform(limits["minx"], limits["maxx"], M).reshape(-1, 1)
    ry = rng.uniform(limits["miny"], limits["maxy"], M).reshape(-1, 1)
    points = gpd.GeoSeries(gpd.points_from_xy(rx, ry, crs="EPSG:3857"))
    indicator = points.within(polygon.iloc[0]).to_numpy().reshape(-1, 1)
    T_total = (gdf["timestamp"].max() - gdf["timestamp"].min()).total_seconds()
    gdf["T_i"] = (gdf["next_time"] - gdf["timestamp"]).dt.total_seconds()
    gdf["hz"] = gdf.apply(
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
        (limits["maxx"] - limits["minx"]) * (limits["maxy"] - limits["miny"]) * gdf["hz"].sum() / M
    )
    res = res / T_total
    del gdf["T_i"]
    del gdf["hz"]
    return res


def norm_pdf(x, mu, variance):
    mu, variance = mu.reshape(-1, 1), variance.reshape(-1, 1)
    x = x.reshape(-1, 1)
    numerator = x - mu
    denominator = 2 * variance
    pdf = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-(numerator**2) / denominator)
    return pdf


def h_z(
    a,
    b,
    T_i,
    x,
    y,
    sigma_m2,
    delta_a,
    delta_b,
    indicator,
    rng,
    n_time_samples=1000,
):
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


def calculate_residence_matrix(df, residence_ageb):
    identifier = str(df["id_adv"].iloc[0])
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=8).digest()
    device_rng = np.random.default_rng(int.from_bytes(digest, "little"))
    gdf = prepare_geometry_data(df)
    sigma_m = calculate_parameters(gdf)
    prob_dist = np.zeros(len(gdf_ageb))
    ageb_count = 0
    for ageb in tqdm(gdf_ageb["CVE_AGEB"].unique(), leave=False):
        prob_dist[ageb_count] = prob_region(gdf_ageb, ageb, gdf, sigma_m, delta_z, 5000, device_rng)
        ageb_count += 1
    return prob_dist.tolist()


def calculate_sigma(df):
    gdf = prepare_geometry_data(df)
    sigma_m = calculate_sigma_m(gdf, delta_z)
    return sigma_m
