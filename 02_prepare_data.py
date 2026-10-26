"""
Step 2 - Download the inventory, finish the feature raster and build the
landslide / non-landslide sample tables.

Usage:
    python 02_prepare_data.py --download-only   # before running 01_gee_export.py
    python 02_prepare_data.py                   # after the GEE GeoTIFF is in data/gee/
Options:
    --skip-osm    do not download OpenStreetMap roads (dist_road is then dropped)
"""
import argparse
import json
import sys
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
from scipy.ndimage import distance_transform_edt

import config as C


# ---------------------------------------------------------------------------
def download_inventory():
    if not C.INVENTORY_ZIP.exists():
        import requests
        print("Downloading inventory from Zenodo ...")
        r = requests.get(C.INVENTORY_URL, timeout=300)
        r.raise_for_status()
        C.INVENTORY_ZIP.write_bytes(r.content)
    with zipfile.ZipFile(C.INVENTORY_ZIP) as z:
        for member in z.namelist():
            name = member.split("/")[-1]
            if name:
                (C.RAW / name).write_bytes(z.read(member))
    print("Inventory files:", sorted(p.name for p in C.RAW.glob("*.shp")))


def read_layer(name):
    gdf = gpd.read_file(C.RAW / name)
    if gdf.crs is None:
        gdf = gdf.set_crs(C.CRS)
    return gdf.to_crs(C.CRS)


# ---------------------------------------------------------------------------
def load_gee_stack():
    tifs = sorted(C.GEE.glob(f"{C.GEE_FILE_PREFIX}*.tif"))
    if not tifs:
        sys.exit(f"No GeoTIFF found in {C.GEE}. Run 01_gee_export.py and copy the export here.")
    srcs = [rasterio.open(t) for t in tifs]
    if len(srcs) == 1:
        arr, transform = srcs[0].read().astype("float32"), srcs[0].transform
    else:  # GEE splits large exports into tiles
        arr, transform = merge(srcs)
        arr = arr.astype("float32")
    profile = srcs[0].profile.copy()
    nodata = srcs[0].nodata
    for s in srcs:
        s.close()
    if arr.shape[0] != len(C.RAW_BANDS):
        sys.exit(f"Expected {len(C.RAW_BANDS)} bands, found {arr.shape[0]}.")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform,
                   count=arr.shape[0], dtype="float32", nodata=np.nan)
    return {b: arr[i] for i, b in enumerate(C.RAW_BANDS)}, profile


def distance_raster(mask, res):
    """Euclidean distance (m) from every pixel to the nearest True pixel."""
    if not mask.any():
        return None
    return (distance_transform_edt(~mask) * res).astype("float32")


def osm_road_mask(aoi, shape, transform):
    import osmnx as ox
    poly = aoi.to_crs(4326).dissolve().geometry.iloc[0]
    print("Downloading OpenStreetMap roads ...")
    roads = ox.features_from_polygon(poly, tags=C.OSM_ROAD_TAGS)
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].to_crs(C.CRS)
    print(f"  {len(roads)} road segments")
    return rasterize(((g, 1) for g in roads.geometry), out_shape=shape,
                     transform=transform, fill=0, all_touched=True).astype(bool)


def build_feature_raster(skip_osm):
    bands, profile = load_gee_stack()
    shape, transform = bands["elevation"].shape, profile["transform"]
    valid = np.isfinite(bands["elevation"])

    stream = np.nan_to_num(bands["upa"], nan=0) >= C.STREAM_UPA_KM2
    bands["dist_stream"] = distance_raster(stream, C.RES)

    bands["dist_road"] = None
    if not skip_osm:
        try:
            aoi = read_layer(C.SHP_AOI)
            bands["dist_road"] = distance_raster(osm_road_mask(aoi, shape, transform), C.RES)
        except Exception as e:  # network problems etc.
            print("  OSM roads skipped:", e)

    names = [b for b in C.CONTINUOUS + C.CATEGORICAL if bands.get(b) is not None]
    dropped = [b for b in C.CONTINUOUS + C.CATEGORICAL if b not in names]
    if dropped:
        print("Features not available and dropped:", dropped)
    stack = np.stack([np.where(valid, bands[b], np.nan) for b in names]).astype("float32")
    profile.update(count=len(names), compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    out = C.PROC / "features.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(stack)
        dst.descriptions = tuple(names)
    (C.PROC / "feature_names.json").write_text(json.dumps(names))
    print(f"Feature raster: {out}  shape={stack.shape}")
    return names


# ---------------------------------------------------------------------------
def sample_raster(xs, ys, names):
    with rasterio.open(C.PROC / "features.tif") as src:
        vals = np.array(list(src.sample(zip(xs, ys))), dtype="float32")
    return pd.DataFrame(vals, columns=names)


def pixel_ids(xs, ys, transform):
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    return np.asarray(rows), np.asarray(cols)


def build_samples(names):
    rng = np.random.default_rng(C.SEED)
    aoi = read_layer(C.SHP_AOI)
    polys = read_layer(C.SHP_POLYGONS)
    pts = read_layer(C.SHP_POINTS)
    pts = pts[pts.within(aoi.dissolve().geometry.iloc[0])]

    with rasterio.open(C.PROC / "features.tif") as src:
        transform, shape = src.transform, (src.height, src.width)
        valid = np.isfinite(src.read(1))

    # --- positives: one sample per landslide pixel ---------------------------
    xs, ys = pts.geometry.x.values, pts.geometry.y.values
    r, c = pixel_ids(xs, ys, transform)
    inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    pos = pd.DataFrame({"x": xs[inside], "y": ys[inside], "r": r[inside], "c": c[inside]})
    pos = pos.drop_duplicates(["r", "c"])
    pos = pos[valid[pos.r, pos.c]]

    # --- negatives: inside AOI, away from any landslide polygon --------------
    aoi_mask = rasterize(((g, 1) for g in aoi.geometry), out_shape=shape,
                         transform=transform, fill=0).astype(bool)
    excl = rasterize(((g, 1) for g in polys.buffer(C.EXCLUSION_BUFFER_M).geometry),
                     out_shape=shape, transform=transform, fill=0, all_touched=True).astype(bool)
    cand = np.flatnonzero(aoi_mask & valid & ~excl)
    n_neg = int(round(len(pos) * C.NEG_RATIO))
    pick = rng.choice(cand, size=min(n_neg, cand.size), replace=False)
    nr, nc = np.unravel_index(pick, shape)
    nx, ny = rasterio.transform.xy(transform, nr, nc)
    neg = pd.DataFrame({"x": np.asarray(nx), "y": np.asarray(ny)})

    pos_df = pd.concat([pos[["x", "y"]].reset_index(drop=True),
                        sample_raster(pos.x, pos.y, names)], axis=1).assign(label=1)
    neg_df = pd.concat([neg, sample_raster(neg.x, neg.y, names)], axis=1).assign(label=0)
    samples = pd.concat([pos_df, neg_df], ignore_index=True)
    samples = samples.dropna(subset=[n for n in names if n not in C.CATEGORICAL])
    samples.to_csv(C.PROC / "samples.csv", index=False)

    # --- dated landslides (for the temporal experiment) ----------------------
    dated = read_layer(C.SHP_DATED)
    ycol = next((col for col in dated.columns if col.lower() == "year"), None)
    if ycol is not None:
        dated = dated[dated.within(aoi.dissolve().geometry.iloc[0])]
        d = pd.concat([pd.DataFrame({"x": dated.geometry.x.values, "y": dated.geometry.y.values,
                                     "year": pd.to_numeric(dated[ycol], errors="coerce").values}),
                       sample_raster(dated.geometry.x.values, dated.geometry.y.values, names)], axis=1)
        d = d.dropna(subset=["year"] + [n for n in names if n not in C.CATEGORICAL])
        d.to_csv(C.PROC / "dated_samples.csv", index=False)
        print(f"Dated landslides: {len(d)}  ({int(d.year.min())}-{int(d.year.max())})")
    else:
        print("No 'Year' field found in the dated shapefile; temporal test will be skipped.")

    summary = {
        "inventory_points_in_aoi": int(len(pts)),
        "positive_samples": int((samples.label == 1).sum()),
        "negative_samples": int((samples.label == 0).sum()),
        "aoi_area_km2": round(float(aoi.dissolve().area.iloc[0]) / 1e6, 1),
        "inventory_polygons": int(len(polys)),
    }
    (C.TABLES / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    study_area_figure(aoi, pts, names)


def study_area_figure(aoi, pts, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with rasterio.open(C.PROC / "features.tif") as src:
        elev = src.read(names.index("elevation") + 1)
        t = src.transform
    ext = [t.c, t.c + t.a * elev.shape[1], t.f + t.e * elev.shape[0], t.f]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(elev, cmap="terrain", extent=ext, interpolation="nearest")
    aoi.boundary.plot(ax=ax, color="black", linewidth=0.7)
    ax.scatter(pts.geometry.x, pts.geometry.y, s=0.3, c="red", alpha=0.5, label="Landslide (inventory)")
    ax.legend(frameon=False, loc="lower left", markerscale=10, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Elevation (m)")
    ax.set_xlabel("Easting (m, UTM 44N)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig_study_area.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-osm", action="store_true")
    a = ap.parse_args()
    if not a.skip_download:
        download_inventory()
    if a.download_only:
        return
    names = build_feature_raster(a.skip_osm)
    build_samples(names)


if __name__ == "__main__":
    main()
