"""
Step 1 - Build the conditioning-factor stack in Google Earth Engine and export
it to Google Drive as a 30 m GeoTIFF in UTM 44N.

Run after 02_prepare_data.py --download-only (it needs the AOI shapefile).

Usage (Colab):
    python 01_gee_export.py
Then wait for the task to finish (https://code.earthengine.google.com/tasks),
and copy the exported GeoTIFF(s) from Drive into data/gee/.
"""
import json

import ee
import geopandas as gpd
from shapely.geometry import mapping

import config as C


def aoi_geometry():
    aoi = gpd.read_file(C.RAW / C.SHP_AOI).to_crs(C.CRS)
    shape = aoi.dissolve().geometry.iloc[0].buffer(C.AOI_BUFFER_M)
    shape = gpd.GeoSeries([shape], crs=C.CRS).to_crs(4326).iloc[0].simplify(0.001)
    return ee.Geometry(json.loads(json.dumps(mapping(shape))))


def build_stack(geom):
    proj = ee.Projection(C.CRS).atScale(C.RES)

    # --- Terrain: Copernicus GLO-30 DEM ---------------------------------
    dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(geom)
           .select("DEM").mosaic().setDefaultProjection(proj).rename("elevation"))
    slope = ee.Terrain.slope(dem).rename("slope")                 # degrees
    aspect_rad = ee.Terrain.aspect(dem).multiply(3.14159265 / 180)
    northness = aspect_rad.cos().rename("northness")
    eastness = aspect_rad.sin().rename("eastness")
    tpi = dem.subtract(dem.focal_mean(radius=300, units="meters")).rename("tpi")
    relief = (dem.focal_max(radius=150, units="meters")
              .subtract(dem.focal_min(radius=150, units="meters")).rename("relief"))
    curvature = dem.convolve(ee.Kernel.laplacian8(normalize=False)).rename("curvature")

    # --- Hydrology: MERIT Hydro (upstream area, height above drainage) ---
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    upa = merit.select("upa").rename("upa")                       # km2
    hnd = merit.select("hnd").rename("hnd")                       # m
    slope_rad = slope.multiply(3.14159265 / 180).max(0.001)
    twi = (upa.multiply(1e6).divide(90).divide(slope_rad.tan())).log().rename("twi")

    # --- Rainfall: CHIRPS daily, monsoon (June-September) ----------------
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(geom)
    years = ee.List.sequence(C.YEAR_START, C.YEAR_END)

    def jjas(y):
        y = ee.Number(y)
        c = (chirps.filter(ee.Filter.calendarRange(y, y, "year"))
             .filter(ee.Filter.calendarRange(6, 9, "month")))
        return c.sum().rename("rain_jjas").addBands(c.max().rename("rain_maxday"))

    rain = ee.ImageCollection.fromImages(years.map(jjas)).mean()

    # --- Vegetation: Landsat 8 post-monsoon NDVI -------------------------
    def mask_l8(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        sr = img.select(["SR_B4", "SR_B5"]).multiply(0.0000275).add(-0.2)
        return sr.updateMask(clear)

    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(geom)
          .filterDate(f"{C.NDVI_YEARS[0]}-01-01", f"{C.NDVI_YEARS[1]}-12-31")
          .filter(ee.Filter.calendarRange(10, 12, "month"))
          .map(mask_l8).median())
    ndvi = l8.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")

    # --- Land cover: ESA WorldCover 2021 ---------------------------------
    landcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")

    # --- Soil: SoilGrids clay content (0-5 cm), g/kg ----------------------
    clay = ee.Image("projects/soilgrids-isric/clay_mean").select("clay_0-5cm_mean").rename("clay")

    stack = ee.Image.cat([dem, slope, northness, eastness, tpi, relief, curvature,
                          twi, upa, hnd, rain, ndvi, landcover, clay])
    return stack.select(C.RAW_BANDS).toFloat().clip(geom)


def main():
    ee.Initialize(project=C.GEE_PROJECT)
    geom = aoi_geometry()
    stack = build_stack(geom)
    print("Bands:", stack.bandNames().getInfo())
    task = ee.batch.Export.image.toDrive(
        image=stack, description=C.GEE_FILE_PREFIX, folder=C.GEE_DRIVE_FOLDER,
        fileNamePrefix=C.GEE_FILE_PREFIX, region=geom, scale=C.RES, crs=C.CRS,
        maxPixels=1e10, fileFormat="GeoTIFF")
    task.start()
    print("Export started. Check progress at https://code.earthengine.google.com/tasks")
    print(f"When finished, copy Drive/{C.GEE_DRIVE_FOLDER}/{C.GEE_FILE_PREFIX}*.tif to {C.GEE}")


if __name__ == "__main__":
    main()
