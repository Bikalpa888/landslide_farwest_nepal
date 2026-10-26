"""
Step 3 - Model experiments.

  E1  Random vs spatial-block cross-validation for LR, RF, XGBoost and MLP
      (repeated 5-fold CV, identical folds for every model)
  E2  Sensitivity of XGBoost performance to the spatial block size
  E3  Temporal transfer: train on landslides dated <= 2010, test on 2011-2018
  E4  SHAP explanation of the final XGBoost model

All numbers used in the paper are written to results/results_for_paper.md.
"""
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             cohen_kappa_score, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config as C

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})

LC_NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
            60: "Bare/sparse", 70: "Snow/ice", 80: "Water", 90: "Wetland", 95: "Mangroves",
            100: "Moss/lichen"}
NICE = {"elevation": "Elevation", "slope": "Slope", "northness": "Northness",
        "eastness": "Eastness", "tpi": "TPI", "relief": "Local relief",
        "curvature": "Curvature", "twi": "TWI", "hnd": "HAND", "rain_jjas": "Monsoon rainfall",
        "rain_maxday": "Max daily rainfall", "ndvi": "NDVI", "clay": "Clay content",
        "dist_stream": "Distance to stream", "dist_road": "Distance to road"}


# ---------------------------------------------------------------------------
def make_models():
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "RF": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                     n_jobs=-1, random_state=C.SEED),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 n_jobs=-1, random_state=C.SEED, tree_method="hist"),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                             early_stopping=True, max_iter=300, random_state=C.SEED)),
    }


def design_matrix(df, names):
    cont = [n for n in names if n in C.CONTINUOUS]
    X = df[cont].copy()
    if "landcover" in names:
        lc = df["landcover"].round().astype("Int64")
        for code in sorted(lc.dropna().unique()):
            X[f"lc_{LC_NAMES.get(int(code), int(code))}"] = (lc == code).astype(float).fillna(0)
    return X


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "Accuracy": accuracy_score(y, yhat), "F1": f1_score(y, yhat),
            "Kappa": cohen_kappa_score(y, yhat), "Brier": brier_score_loss(y, p)}


def random_folds(y, seed):
    skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=seed)
    fold = np.zeros(len(y), int)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[te] = k
    return fold


def block_folds(x, y_coord, size, seed):
    """Assign square spatial blocks (size x size m) to folds at random."""
    bx, by = np.floor(x / size).astype(int), np.floor(y_coord / size).astype(int)
    block = pd.factorize(pd.Series(bx).astype(str) + "_" + pd.Series(by).astype(str))[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(block.max() + 1)
    return order[block] % C.N_FOLDS, block.max() + 1


def run_cv(X, y, folds, model_name):
    out = []
    for k in range(C.N_FOLDS):
        tr, te = folds != k, folds == k
        if len(np.unique(y[te])) < 2:
            continue
        m = make_models()[model_name]
        m.fit(X[tr], y[tr])
        out.append({"fold": k, **metrics(y[te], m.predict_proba(X[te])[:, 1])})
    return out


def fmt(mean, sd, d=3):
    return f"{mean:.{d}f} ± {sd:.{d}f}"


# ---------------------------------------------------------------------------
def experiment_cv(df, X, y, report):
    rows = []
    for rep in range(C.N_REPEATS):
        seed = C.SEED + rep
        schemes = {"Random": random_folds(y, seed),
                   "Spatial": block_folds(df.x.values, df.y.values, C.BLOCK_SIZE_M, seed)[0]}
        for scheme, folds in schemes.items():
            for name in make_models():
                t0 = time.time()
                for r in run_cv(X, y, folds, name):
                    rows.append({"repeat": rep, "scheme": scheme, "model": name, **r})
                print(f"  repeat {rep + 1}/{C.N_REPEATS}  {scheme:<7} {name:<8} {time.time() - t0:5.1f}s")
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "E1_cv_folds.csv", index=False)

    summ = res.groupby(["model", "scheme"]).agg(["mean", "std"]).drop(columns=["repeat", "fold"])
    summ.to_csv(C.TABLES / "E1_cv_summary.csv")

    # Table for the paper
    lines = ["| Model | Validation | AUC | PR-AUC | Accuracy | F1 | Kappa |",
             "|---|---|---|---|---|---|---|"]
    order = ["LR", "RF", "XGBoost", "MLP"]
    for m in order:
        for s in ["Random", "Spatial"]:
            r = summ.loc[(m, s)]
            lines.append(f"| {m} | {s} | " + " | ".join(
                fmt(r[(k, 'mean')], r[(k, 'std')]) for k in ["AUC", "PR_AUC", "Accuracy", "F1", "Kappa"]) + " |")
    report.append("## Table 3. Random vs spatial cross-validation (mean ± SD over "
                  f"{C.N_REPEATS}x{C.N_FOLDS} folds)\n\n" + "\n".join(lines))

    # Optimism gap + Wilcoxon test
    gap_lines = ["| Model | AUC random | AUC spatial | Gap (random − spatial) | Wilcoxon p |",
                 "|---|---|---|---|---|"]
    for m in order:
        a = res[(res.model == m) & (res.scheme == "Random")].sort_values(["repeat", "fold"]).AUC.values
        b = res[(res.model == m) & (res.scheme == "Spatial")].sort_values(["repeat", "fold"]).AUC.values
        n = min(len(a), len(b))
        p = wilcoxon(a[:n], b[:n]).pvalue if n >= 5 else float("nan")
        gap_lines.append(f"| {m} | {a.mean():.3f} | {b.mean():.3f} | {a.mean() - b.mean():.3f} | {p:.2g} |")
    report.append("## Table 4. Optimism gap in AUC\n\n" + "\n".join(gap_lines))

    best = summ.xs("Spatial", level="scheme")[("AUC", "mean")].idxmax()
    report.append(f"Best model under spatial CV: **{best}**")

    # Figure: AUC by model and scheme
    fig, ax = plt.subplots(figsize=(6, 3.4))
    pos = np.arange(len(order))
    for i, (s, col) in enumerate([("Random", "#9e9e9e"), ("Spatial", "#1f4e79")]):
        data = [res[(res.model == m) & (res.scheme == s)].AUC.values for m in order]
        bp = ax.boxplot(data, positions=pos + (i - 0.5) * 0.35, widths=0.3, patch_artist=True,
                        medianprops=dict(color="black"))
        for b in bp["boxes"]:
            b.set_facecolor(col)
        ax.plot([], [], color=col, lw=8, label=f"{s} CV")
    ax.set_xticks(pos, order)
    ax.set_ylabel("AUC (test folds)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig_auc_random_vs_spatial.png")
    plt.close(fig)
    return best


def experiment_blocksize(df, X, y, report):
    rows = []
    for size in C.BLOCK_SIZES_M:
        for rep in range(C.N_REPEATS):
            folds, nblocks = block_folds(df.x.values, df.y.values, size, C.SEED + rep)
            for r in run_cv(X, y, folds, "XGBoost"):
                rows.append({"block_km": size / 1000, "repeat": rep, "n_blocks": nblocks, **r})
        print(f"  block {size / 1000:g} km done")
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "E2_blocksize.csv", index=False)
    g = res.groupby("block_km").AUC.agg(["mean", "std"]).reset_index()
    nb = res.groupby("block_km").n_blocks.first()
    lines = ["| Block size (km) | Number of blocks | AUC (mean ± SD) |", "|---|---|---|"]
    for _, r in g.iterrows():
        lines.append(f"| {r.block_km:g} | {nb[r.block_km]} | {fmt(r['mean'], r['std'])} |")
    report.append("## Table 5. XGBoost AUC vs spatial block size\n\n" + "\n".join(lines))

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(g.block_km, g["mean"], yerr=g["std"], marker="o", color="#1f4e79", capsize=3)
    ax.set_xscale("log")
    ax.set_xticks(g.block_km, [f"{v:g}" for v in g.block_km])
    ax.set_xlabel("Spatial block size (km)")
    ax.set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig_blocksize.png")
    plt.close(fig)


def experiment_temporal(df, names, report):
    path = C.PROC / "dated_samples.csv"
    if not path.exists():
        report.append("## Temporal test\n\nSkipped (no dated landslides).")
        return
    d = pd.read_csv(path)
    neg = df[df.label == 0].sample(frac=1, random_state=C.SEED)
    old, new = d[d.year <= C.TEMPORAL_SPLIT_YEAR], d[d.year > C.TEMPORAL_SPLIT_YEAR]
    if len(old) < 50 or len(new) < 50:
        report.append("## Temporal test\n\nSkipped (too few dated landslides in one period).")
        return
    # split negatives in proportion so that train/test keep a 1:1 balance
    n_tr = min(len(old), int(len(neg) * len(old) / (len(old) + len(new))))
    neg_tr, neg_te = neg.iloc[:n_tr], neg.iloc[n_tr:n_tr + len(new)]
    train = pd.concat([old.assign(label=1), neg_tr])
    test = pd.concat([new.assign(label=1), neg_te])
    cols = design_matrix(pd.concat([train, test]), names).columns
    Xtr = design_matrix(train, names).reindex(columns=cols, fill_value=0).values
    Xte = design_matrix(test, names).reindex(columns=cols, fill_value=0).values
    lines = ["| Model | AUC | PR-AUC | F1 |", "|---|---|---|---|"]
    for name, m in make_models().items():
        m.fit(Xtr, train.label.values)
        r = metrics(test.label.values, m.predict_proba(Xte)[:, 1])
        lines.append(f"| {name} | {r['AUC']:.3f} | {r['PR_AUC']:.3f} | {r['F1']:.3f} |")
    report.append(f"## Table 6. Temporal transfer (train: landslides ≤ {C.TEMPORAL_SPLIT_YEAR}, "
                  f"n = {len(old)}; test: {C.TEMPORAL_SPLIT_YEAR + 1}–{int(d.year.max())}, n = {len(new)})\n\n"
                  + "\n".join(lines))
    # landslides per year (figure)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    d.year.astype(int).value_counts().sort_index().plot.bar(ax=ax, color="#1f4e79", width=0.8)
    ax.set_xlabel("Monsoon year")
    ax.set_ylabel("Dated landslides")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    fig.tight_layout()
    fig.savefig(C.FIGS / "fig_dated_per_year.png")
    plt.close(fig)


def experiment_shap(X, y, report):
    import shap
    model = make_models()["XGBoost"].fit(X.values, y)
    idx = np.random.default_rng(C.SEED).choice(len(X), min(C.SHAP_SAMPLE, len(X)), replace=False)
    Xs = X.iloc[idx]
    sv = shap.TreeExplainer(model).shap_values(Xs.values)
    disp = [NICE.get(c, c.replace("lc_", "LC: ")) for c in X.columns]
    imp = pd.DataFrame({"feature": disp, "mean_abs_shap": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(C.TABLES / "E4_shap_importance.csv", index=False)
    lines = ["| Rank | Factor | Mean |SHAP| |", "|---|---|---|"]
    for i, r in enumerate(imp.head(10).itertuples(), 1):
        lines.append(f"| {i} | {r.feature} | {r.mean_abs_shap:.3f} |")
    report.append("## Table 7. Top 10 factors by mean |SHAP| (XGBoost)\n\n" + "\n".join(lines))
    plt.figure()
    shap.summary_plot(sv, Xs.values, feature_names=disp, max_display=12, show=False)
    plt.gcf().set_size_inches(6.5, 4.6)
    plt.tight_layout()
    plt.savefig(C.FIGS / "fig_shap_summary.png")
    plt.close("all")


def describe_factors(df, names, report):
    cont = [n for n in names if n in C.CONTINUOUS]
    t = df.groupby("label")[cont].median().T
    t.columns = ["Non-landslide (median)", "Landslide (median)"]
    lines = ["| Factor | Landslide (median) | Non-landslide (median) |", "|---|---|---|"]
    for f, r in t.iterrows():
        lines.append(f"| {NICE.get(f, f)} | {r['Landslide (median)']:.2f} | {r['Non-landslide (median)']:.2f} |")
    report.append("## Table 2. Factor medians by class\n\n" + "\n".join(lines))
    corr = df[cont].corr(method="spearman").abs()
    high = [(a, b, corr.loc[a, b]) for i, a in enumerate(cont) for b in cont[i + 1:] if corr.loc[a, b] > 0.8]
    report.append("Highly correlated pairs (|Spearman| > 0.8): " +
                  (", ".join(f"{NICE.get(a, a)}–{NICE.get(b, b)} ({v:.2f})" for a, b, v in high) or "none"))


# ---------------------------------------------------------------------------
def main():
    names = json.loads((C.PROC / "feature_names.json").read_text())
    df = pd.read_csv(C.PROC / "samples.csv")
    X = design_matrix(df, names)
    y = df.label.values
    report = ["# Results for the paper (generated by 03_experiments.py)",
              f"Samples: {int(y.sum())} landslide, {int((1 - y).sum())} non-landslide; "
              f"features used: {len(X.columns)} ({', '.join(X.columns)})"]
    summary_path = C.TABLES / "data_summary.json"
    if summary_path.exists():
        report.append("Data summary: " + summary_path.read_text())

    describe_factors(df, names, report)
    print("E1: random vs spatial CV")
    best = experiment_cv(df, X.values, y, report)
    print("E2: block-size sensitivity")
    experiment_blocksize(df, X.values, y, report)
    print("E3: temporal transfer")
    experiment_temporal(df, names, report)
    print("E4: SHAP")
    experiment_shap(X, y, report)
    (C.RESULTS / "results_for_paper.md").write_text("\n\n".join(report), encoding="utf-8")
    print(f"\nDone. Best spatial-CV model: {best}. See results/results_for_paper.md")


if __name__ == "__main__":
    main()
