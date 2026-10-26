"""
Step 4 - Train the final model on all samples, predict susceptibility for every
30 m pixel, classify it into five classes and check how many inventory
landslides fall into each class. Optionally compare with the national model.
"""
import importlib
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

import config as C

exp = importlib.import_module("03_experiments")
CLASSES = ["Very low", "Low", "Moderate", "High", "Very high"]


def main():
    names = json.loads((C.PROC / "feature_names.json").read_text())
    df = pd.read_csv(C.PROC / "samples.csv")
    X = exp.design_matrix(df, names)
    cols = list(X.columns)
    model = exp.make_models()[C.MAP_MODEL].fit(X.values, df.label.values)

    src = rasterio.open(C.PROC / "features.tif")
    prof = src.profile.copy()
    prof.update(count=1, dtype="float32", nodata=np.nan)
    out_path = C.PROC / "susceptibility.tif"
    with rasterio.open(out_path, "w", **prof) as dst:
        step = 512
        for row in range(0, src.height, step):
            win = Window(0, row, src.width, min(step, src.height - row))
            arr = src.read(window=win).reshape(src.count, -1).T
            ok = np.isfinite(arr[:, [i for i, n in enumerate(names) if n != "landcover"]]).all(1)
            pred = np.full(arr.shape[0], np.nan, dtype="float32")
            if ok.any():
                block = exp.design_matrix(pd.DataFrame(arr[ok], columns=names), names)
                block = block.reindex(columns=cols, fill_value=0)
                pred[ok] = model.predict_proba(block.values)[:, 1]
            dst.write(pred.reshape(int(win.height), int(win.width)), 1, window=win)
    transform, shape = src.transform, (src.height, src.width)
    src.close()

    with rasterio.open(out_path) as s:
        susc = s.read(1)
    aoi = gpd.read_file(C.RAW / C.SHP_AOI).to_crs(C.CRS)
    aoi_mask = rasterize(((g, 1) for g in aoi.geometry), out_shape=shape,
                         transform=transform, fill=0).astype(bool)
    susc = np.where(aoi_mask, susc, np.nan)
    valid = np.isfinite(susc)

    # five equal-area (quantile) classes
    breaks = np.nanquantile(susc, [0.2, 0.4, 0.6, 0.8])
    cls = np.full(shape, -1, int)
    cls[valid] = np.digitize(susc[valid], breaks)

    # landslide pixels per class (all inventory polygons)
    polys = gpd.read_file(C.RAW / C.SHP_POLYGONS).to_crs(C.CRS)
    ls = rasterize(((g, 1) for g in polys.geometry), out_shape=shape, transform=transform,
                   fill=0, all_touched=False).astype(bool) & valid
    rows = []
    for k, name in enumerate(CLASSES):
        area = (cls == k).sum() / valid.sum() * 100
        lsp = (ls & (cls == k)).sum() / max(ls.sum(), 1) * 100
        rows.append({"Class": name, "Area (%)": area, "Landslide area (%)": lsp,
                     "Frequency ratio": lsp / area if area else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(C.TABLES / "E5_classes.csv", index=False)
    lines = ["| Class | Area (%) | Landslide area (%) | Frequency ratio |", "|---|---|---|---|"]
    for r in t.itertuples():
        lines.append(f"| {r.Class} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.2f} |")
    report = [f"## Table 8. Susceptibility classes ({C.MAP_MODEL}, quantile breaks: "
              + ", ".join(f"{b:.3f}" for b in breaks) + ")\n\n" + "\n".join(lines)]

    # optional comparison with the national 30 m model
    nat = sorted(C.NATIONAL_DIR.glob("*.tif")) if C.NATIONAL_DIR.exists() else []
    if nat:
        with rasterio.open(nat[0]) as ns:
            dst_arr = np.full(shape, np.nan, dtype="float32")
            reproject(ns.read(1).astype("float32"), dst_arr, src_transform=ns.transform,
                      src_crs=ns.crs, src_nodata=ns.nodata, dst_transform=transform, dst_crs=C.CRS,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        both = valid & np.isfinite(dst_arr)
        rng = np.random.default_rng(C.SEED)
        idx = rng.choice(np.flatnonzero(both), min(200000, both.sum()), replace=False)
        rho = pd.Series(susc.ravel()[idx]).corr(pd.Series(dst_arr.ravel()[idx]), method="spearman")
        report.append(f"Spearman correlation with national 30 m model (n = {idx.size} pixels): {rho:.3f}")

    with open(C.RESULTS / "results_for_paper.md", "a", encoding="utf-8") as f:
        f.write("\n\n" + "\n\n".join(report))

    # map figure
    ext = [transform.c, transform.c + transform.a * shape[1],
           transform.f + transform.e * shape[0], transform.f]
    cmap = ListedColormap(["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(np.ma.masked_less(cls, 0), cmap=cmap, extent=ext, interpolation="nearest")
    aoi.boundary.plot(ax=ax, color="black", linewidth=0.6)
    for k, name in enumerate(CLASSES):
        ax.plot([], [], "s", color=cmap(k), markersize=10, label=name)
    ax.legend(title="Susceptibility", frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.set_xlabel("Easting (m, UTM 44N)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig_susceptibility_map.png", dpi=300)
    plt.close(fig)
    print("\n".join(report))


if __name__ == "__main__":
    main()
