"""
Compare a fresh run with the values published in the paper.

Usage:  python verify_results.py
Prints PASS/FAIL for each published number (tolerance 0.002 AUC).
"""
import sys

import pandas as pd

import config as C

TOL = 0.002

# Published values (Dhungana, Tables 3 and 5)
PUBLISHED_E1 = {
    ("LR", "Random"): 0.7954, ("LR", "Spatial"): 0.7941,
    ("RF", "Random"): 0.8368, ("RF", "Spatial"): 0.8215,
    ("XGBoost", "Random"): 0.8460, ("XGBoost", "Spatial"): 0.8248,
    ("MLP", "Random"): 0.8253, ("MLP", "Spatial"): 0.8178,
}
PUBLISHED_E2 = {1.0: 0.8365, 2.5: 0.8317, 5.0: 0.8248, 10.0: 0.8201, 20.0: 0.8141}
PUBLISHED_SAMPLES = (25280, 25122)


def main():
    ok = True
    df = pd.read_csv(C.PROC / "samples.csv")
    n = (int((df.label == 1).sum()), int((df.label == 0).sum()))
    good = n == PUBLISHED_SAMPLES
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  samples (landslide, non-landslide) = {n}, published {PUBLISHED_SAMPLES}")

    s = pd.read_csv(C.TABLES / "E1_cv_summary.csv", header=[0, 1], index_col=[0, 1])
    for key, pub in PUBLISHED_E1.items():
        new = float(s.loc[key, ("AUC", "mean")])
        good = abs(new - pub) <= TOL
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  E1 {key[0]:<8} {key[1]:<8} AUC {new:.4f}  (published {pub:.4f})")

    b = pd.read_csv(C.TABLES / "E2_blocksize.csv").groupby("block_km").AUC.mean()
    for km, pub in PUBLISHED_E2.items():
        new = float(b.loc[km])
        good = abs(new - pub) <= TOL
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  E2 block {km:>4g} km  AUC {new:.4f}  (published {pub:.4f})")

    print("\nAll published values reproduced." if ok else "\nSome values differ - check the settings in config.py.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
