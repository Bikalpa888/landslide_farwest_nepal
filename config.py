"""
Central settings for the Far-Western Nepal landslide susceptibility study.
Edit only this file when you change paths, the study area or experiment settings.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# LS_WORKDIR lets the test suite run in a separate folder
ROOT = Path(os.environ.get("LS_WORKDIR", Path(__file__).resolve().parent))
DATA = ROOT / "data"
RAW = DATA / "raw"                # downloaded inventory (zip + shapefiles)
GEE = DATA / "gee"                # GeoTIFF(s) exported from Google Earth Engine
PROC = DATA / "processed"         # feature raster + sample tables
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"

for p in (RAW, GEE, PROC, TABLES, FIGS):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Landslide inventory (Munoz-Torrero Manchado et al., 2021) on Zenodo
# ---------------------------------------------------------------------------
INVENTORY_URL = (
    "https://zenodo.org/records/4290100/files/"
    "LandslideInventory_FarWesternNepal.zip?download=1"
)
INVENTORY_ZIP = RAW / "LandslideInventory_FarWesternNepal.zip"
# File names inside the zip (as documented on the Zenodo record)
SHP_POLYGONS = "LandslideInventory_FarWesternNepal_Pol.shp"
SHP_POINTS = "LandslideInventory_FarWesternNepal_Points.shp"
SHP_DATED = "LandslideInventory_FarWesternNepal_Points_Dated1992_2018.shp"
SHP_AOI = "LandslideInventory_FarWesternNepal_AOI.shp"

# ---------------------------------------------------------------------------
# Spatial reference and grid
# ---------------------------------------------------------------------------
CRS = "EPSG:32644"     # WGS 84 / UTM zone 44N (same as the inventory)
RES = 30               # metres
AOI_BUFFER_M = 1000    # buffer around the AOI for the GEE export (edge effects)

# ---------------------------------------------------------------------------
# Google Earth Engine
# ---------------------------------------------------------------------------
GEE_PROJECT = "YOUR-GEE-CLOUD-PROJECT-ID"   # <- change this
GEE_DRIVE_FOLDER = "landslide_farwest"
GEE_FILE_PREFIX = "fw_stack"
YEAR_START, YEAR_END = 1992, 2018           # inventory period
NDVI_YEARS = (2014, 2018)                   # Landsat 8 available from 2013

# Band order of the GEE export. Do not change unless you edit 01_gee_export.py.
RAW_BANDS = [
    "elevation", "slope", "northness", "eastness", "tpi", "relief",
    "curvature", "twi", "upa", "hnd", "rain_jjas", "rain_maxday",
    "ndvi", "landcover", "clay",
]

# Features used by the models (upa is only used to derive dist_stream)
CONTINUOUS = [
    "elevation", "slope", "northness", "eastness", "tpi", "relief",
    "curvature", "twi", "hnd", "rain_jjas", "rain_maxday", "ndvi", "clay",
    "dist_stream", "dist_road",
]
CATEGORICAL = ["landcover"]
STREAM_UPA_KM2 = 1.0   # upstream area threshold that defines a stream channel

# OpenStreetMap road classes used for distance to road
OSM_ROAD_TAGS = {
    "highway": ["motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "track"]
}

# ---------------------------------------------------------------------------
# Sampling and experiments
# ---------------------------------------------------------------------------
SEED = 42
NEG_RATIO = 1.0             # non-landslide : landslide samples
EXCLUSION_BUFFER_M = 100    # no negatives within this distance of a landslide
N_FOLDS = 5
N_REPEATS = 5               # repeated CV (5 x 5 = 25 folds per scheme)
BLOCK_SIZE_M = 5000         # main spatial block size
BLOCK_SIZES_M = [1000, 2500, 5000, 10000, 20000]   # sensitivity analysis
TEMPORAL_SPLIT_YEAR = 2010  # train <= 2010, test 2011-2018
SHAP_SAMPLE = 5000
MAP_MODEL = "XGBoost"       # model used for the final susceptibility map

# Optional: national 30 m susceptibility model (Zenodo record 8307964).
# Unzip it into this folder; 04_map.py uses the first .tif it finds there.
NATIONAL_DIR = DATA / "national"
