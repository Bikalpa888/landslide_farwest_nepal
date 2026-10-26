# Random versus Spatial Cross-Validation in Landslide Susceptibility Mapping (Far-Western Nepal)

Code for the paper by **Bikalpa Dhungana** (Virinchi College / Asia e University, Kathmandu).

* Processed data: https://doi.org/10.5281/zenodo.23666655
* Code: https://github.com/Bikalpa888/landslide_farwest_nepal

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bikalpa888/landslide_farwest_nepal/blob/main/reproduce_results.ipynb)

The study compares random and spatial-block cross-validation for logistic
regression, random forest, XGBoost and a multilayer perceptron on a published
inventory of 26,350 landslides in Far-Western Nepal (25,280 landslide and
25,122 non-landslide samples, 14 conditioning factors). It also tests the effect
of block size, transfer from 1992-2010 to 2011-2018 landslides, explains the
model with SHAP, and produces a 30 m susceptibility map.

## Data (all open)

| Data | Source |
|---|---|
| Landslide inventory, Far-Western Nepal (1992-2018) | Muñoz-Torrero Manchado et al. (2021), Zenodo record 4290100 |
| Copernicus GLO-30 DEM | Earth Engine `COPERNICUS/DEM/GLO30` |
| MERIT Hydro (upstream area, HAND) | Earth Engine `MERIT/Hydro/v1_0_1` |
| CHIRPS daily rainfall | Earth Engine `UCSB-CHG/CHIRPS/DAILY` |
| Landsat 8 Collection 2 L2 (NDVI) | Earth Engine `LANDSAT/LC08/C02/T1_L2` |
| ESA WorldCover 2021 | Earth Engine `ESA/WorldCover/v200` |
| SoilGrids clay | Earth Engine `projects/soilgrids-isric/clay_mean` |
| Roads (optional; not used in the paper) | OpenStreetMap (via osmnx) |
| National 30 m susceptibility (optional comparison) | Zenodo record 8307964 |

## How to reproduce the results (recommended, no Earth Engine needed)

1. Click the **Open in Colab** badge above (or open `reproduce_results.ipynb` in Google Colab).
2. Click **Runtime → Run all** (the repository and data links are already filled in).
3. Wait (about 90 minutes on the free CPU).
4. The last cell (`verify_results.py`) prints PASS for every published number.

The processed data archive contains `samples.csv`, `dated_samples.csv`,
`feature_names.json` and `features.tif`, exactly as used in the paper.
All random seeds are fixed in `config.py`, so a re-run gives the same values.

## How to rebuild everything from raw data

Open `run_in_colab.ipynb` and follow the cells. It downloads the inventory from
Zenodo, exports the factor stack with Google Earth Engine (needs a registered
non-commercial Cloud project with the Earth Engine API and Google Drive API
enabled), builds the samples with `--skip-osm` (as in the paper), and runs all
experiments. A rebuilt factor stack can differ slightly from the published one if
Earth Engine datasets are updated (for example, `COPERNICUS/DEM/GLO30` has been
superseded by `GLO30_2024_1`), so use the archived data for exact reproduction.

## Experiments

| ID | What it tests |
|---|---|
| E1 | Random vs 5 km spatial-block CV, 5 x 5 repeated folds, LR / RF / XGBoost / MLP, Wilcoxon test on the AUC gap |
| E2 | XGBoost AUC for block sizes of 1, 2.5, 5, 10 and 20 km |
| E3 | Temporal transfer: train on landslides dated 1992-2010, test on 2011-2018 |
| E4 | SHAP importance and dependence for XGBoost |
| E5 | Susceptibility classes (quantile breaks) and frequency ratio of mapped landslide area |

`MAP_MODEL` in `config.py` sets the model used for the map (XGBoost in the paper).

## Software test

`tests/make_synthetic.py` creates a small **fake** dataset so the code can be
tested offline. Results from it are meaningless and must never be reported.

```bash
LS_WORKDIR=/tmp/ls_test python tests/make_synthetic.py
LS_WORKDIR=/tmp/ls_test python 02_prepare_data.py --skip-download --skip-osm
LS_WORKDIR=/tmp/ls_test python 03_experiments.py
LS_WORKDIR=/tmp/ls_test python 04_map.py
```

## Known limitations

* WorldCover (2021) and OSM roads describe present-day conditions, not the 1992-2018 inventory period.
* CHIRPS rainfall (~5 km) is much coarser than the 30 m grid.
* Geology is not included because no open digital geological map of the study area is available in Earth Engine.
* Landslide points mark the centre of the rupture area; non-landslide points are sampled at random outside a 100 m buffer.

## Licence

Code: MIT. Data: see the licence of each source; cite the inventory paper whenever you use it.
