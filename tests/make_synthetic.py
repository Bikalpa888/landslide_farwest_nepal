"""
SOFTWARE TEST ONLY - creates a small FAKE dataset with the same file layout as
the real inputs, so the pipeline can be tested end to end without internet.
Nothing produced from this data may ever be reported as a result.

Usage:
    LS_WORKDIR=/tmp/ls_test python tests/make_synthetic.py
    LS_WORKDIR=/tmp/ls_test python 02_prepare_data.py --skip-download --skip-osm
    LS_WORKDIR=/tmp/ls_test python 03_experiments.py
    LS_WORKDIR=/tmp/ls_test python 04_map.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

rng = np.random.default_rng(0)
H = W = 600
x0, y0 = 500000.0, 3250000.0
T = from_origin(x0, y0, C.RES, C.RES)

field = lambda s: gaussian_filter(rng.normal(size=(H, W)), s)
elev = 1500 + 4000 * field(40)
gy, gx = np.gradient(elev, C.RES)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
asp = np.arctan2(-gx, gy)
bands = {
    "elevation": elev, "slope": slope, "northness": np.cos(asp), "eastness": np.sin(asp),
    "tpi": 50 * field(5), "relief": 100 + 30 * field(8), "curvature": field(3),
    "twi": 6 + 2 * field(6), "upa": np.abs(30 * field(4)), "hnd": np.abs(80 * field(10)),
    "rain_jjas": 1500 + 3000 * field(60), "rain_maxday": 80 + 200 * field(60),
    "ndvi": 0.5 + 3 * field(15), "landcover": rng.choice([10, 20, 30, 40, 60], size=(H, W)).astype(float),
    "clay": 200 + 300 * field(30),
}
stack = np.stack([bands[b] for b in C.RAW_BANDS]).astype("float32")
prof = dict(driver="GTiff", height=H, width=W, count=len(C.RAW_BANDS), dtype="float32",
            crs=C.CRS, transform=T, nodata=-9999)
with rasterio.open(C.GEE / f"{C.GEE_FILE_PREFIX}.tif", "w", **prof) as dst:
    dst.write(stack)

# landslide probability: slope + rainfall + a spatially clustered nuisance term
logit = (0.12 * (slope - slope.mean()) + 1.5 * (bands["rain_jjas"] - 1500) / 1500
         + 4 * field(25) * 20 - 4)
prob = 1 / (1 + np.exp(-logit))
cand = np.flatnonzero(rng.random(H * W) < prob.ravel() * 0.25)
r, c = np.unravel_index(cand, (H, W))
xs, ys = rasterio.transform.xy(T, r, c)
pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=C.CRS)
aoi = gpd.GeoDataFrame(geometry=[box(x0 + 600, y0 - H * C.RES + 600, x0 + W * C.RES - 600, y0 - 600)], crs=C.CRS)
pts = pts[pts.within(aoi.geometry.iloc[0])].reset_index(drop=True)
polys = gpd.GeoDataFrame(geometry=pts.buffer(25), crs=C.CRS)
dated = pts.sample(frac=0.35, random_state=0).copy()
dated["Year"] = rng.integers(1992, 2019, len(dated))

for gdf, name in [(aoi, C.SHP_AOI), (polys, C.SHP_POLYGONS), (pts, C.SHP_POINTS), (dated, C.SHP_DATED)]:
    gdf.to_file(C.RAW / name)
print(f"Synthetic test data written to {C.ROOT} ({len(pts)} fake landslides)")
