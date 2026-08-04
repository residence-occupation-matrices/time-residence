import geopandas as gpd
import pandas as pd
from pandarallel import pandarallel
from project_paths import DATA_ROOT, prepare_output
from tqdm import tqdm

tqdm.pandas()
pandarallel.initialize()


def calculate_polygon(point, gdf_ageb):
    for i, pol in enumerate(gdf_ageb["geometry"]):
        if pol.contains(point):
            return i
    return -1


for fileno in [1, 2, 3]:
    filename = DATA_ROOT / f"M{fileno}.csv"
    chunksize = 20000
    gdf_ageb = gpd.read_file(DATA_ROOT / "geometry" / "26a.shp")
    gdf_ageb.to_crs("EPSG:3857", inplace=True)
    gdf_ageb.sort_values(by="CVE_AGEB", inplace=True)

    result = pd.DataFrame([], dtype=int)
    for chunk in tqdm(pd.read_csv(filename, chunksize=chunksize, sep=";")):
        chunk["timestamp"] = chunk["timestamp"].astype("datetime64[ns, UTC]")
        chunk.drop_duplicates(subset=["id_adv", "timestamp"], keep="first", inplace=True)
        gdf = gpd.GeoDataFrame(
            chunk, geometry=gpd.points_from_xy(chunk["lon"], chunk["lat"], crs="EPSG:4326")
        )
        gdf.to_crs("EPSG:3857", inplace=True)
        gdf["polygon"] = gdf["geometry"].parallel_apply(calculate_polygon, args=(gdf_ageb,))
        result = pd.concat([result, gdf[["id_adv", "timestamp", "lat", "lon", "polygon"]]])

    result.to_csv(
        prepare_output(DATA_ROOT / f"ageb_M{fileno}_sorted.csv.zip"),
        index=False,
        compression="zip",
        header=["id", "timestamp", "lat", "lon", "polygon"],
    )
