"""Preprocess CICIDS2017 (MachineLearningCVE) CSV files.

Steps: load -> clean column names -> drop duplicate columns -> replace inf/NaN
-> drop duplicate rows -> binary label (BENIGN=0, attack=1) -> stratified
train/test split -> StandardScaler (fit on train only) -> save.
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_csvs(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {input_dir}. See data/README.md")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()          # CICIDS headers have leading spaces
        df = df.loc[:, ~df.columns.duplicated()]      # some files repeat a column
        frames.append(df)
        print(f"loaded {os.path.basename(f)}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def clean(df):
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    print(f"cleaning: {n0:,} -> {len(df):,} rows")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/raw")
    ap.add_argument("--output-dir", default="data/processed")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="stratified subsample for low-memory laptops, e.g. 300000")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = clean(load_csvs(args.input_dir))

    label_raw = df["Label"].astype(str).str.strip()
    y_multi = label_raw
    y_bin = (label_raw != "BENIGN").astype(int)
    X = df.drop(columns=["Label"]).select_dtypes(include=[np.number]).astype(np.float32)

    if args.max_rows and len(X) > args.max_rows:
        X, _, y_bin, _, y_multi, _ = train_test_split(
            X, y_bin, y_multi, train_size=args.max_rows,
            stratify=y_bin, random_state=args.seed)
        print(f"subsampled to {len(X):,} rows")

    X_tr, X_te, yb_tr, yb_te, ym_tr, ym_te = train_test_split(
        X, y_bin, y_multi, test_size=args.test_size,
        stratify=y_bin, random_state=args.seed)

    scaler = StandardScaler().fit(X_tr)               # fit on train only (no leakage)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.output_dir, "data.npz"),
                        X_train=X_tr_s, X_test=X_te_s,
                        y_train=yb_tr.values, y_test=yb_te.values,
                        y_train_multi=ym_tr.values, y_test_multi=ym_te.values)
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    with open(os.path.join(args.output_dir, "info.json"), "w") as f:
        json.dump({"features": list(X.columns),
                   "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                   "attack_fraction_train": float(yb_tr.mean()),
                   "classes": sorted(label_raw.unique().tolist())}, f, indent=2)
    print(f"saved to {args.output_dir}: train={len(X_tr):,} test={len(X_te):,} "
          f"features={X.shape[1]} attack%={yb_tr.mean()*100:.1f}")


if __name__ == "__main__":
    main()
