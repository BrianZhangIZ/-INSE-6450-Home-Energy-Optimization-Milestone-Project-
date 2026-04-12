import os
import re
import glob
import numpy as np
import pandas as pd

RAW_GLOB = "data/raw/CLEAN_House*.csv"   # will match House1..House20 etc.
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Settings ----------
RESAMPLE_MIN = 5
HORIZON_H = 12               # next 1 hour at 5-min
LAGS = [1, 2, 3, 6, 12, 24]
SEASONAL_LAG = 288           # 24h ago at 5-min
ROLL_WINDOWS = [12, 72, 288] # 1h, 6h, 24h
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

def cyclical_time_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = dt_index.hour + dt_index.minute / 60.0
    dow = dt_index.dayofweek
    out = pd.DataFrame(index=dt_index)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend"] = (dow >= 5).astype(int)
    return out

def parse_house_id(path: str) -> int:
    m = re.search(r"House(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse house id from filename: {path}")
    return int(m.group(1))

def build_XY_for_house(csv_path: str):
    house_id = parse_house_id(csv_path)

    usecols = ["Unix", "Aggregate", "Issues"]
    df = pd.read_csv(csv_path, usecols=usecols)

    df["dt"] = pd.to_datetime(df["Unix"], unit="s", utc=True)
    df = df.sort_values("dt").set_index("dt")

    # Resample to fixed interval
    rule = f"{RESAMPLE_MIN}min"
    agg = df["Aggregate"].resample(rule).mean()
    issues = df["Issues"].resample(rule).max()

    data = pd.concat([agg.rename("Aggregate"), issues.rename("Issues")], axis=1)

    # implicit gaps (NaNs caused by resample)
    data["gap_flag"] = data["Aggregate"].isna().astype(int)

    # fill gaps simply (good enough for milestone; document it)
    data["Aggregate"] = data["Aggregate"].ffill().bfill()
    data["Issues"] = data["Issues"].fillna(0)

    # Feature blocks
    feats = []
    feats.append(cyclical_time_features(data.index))
    feats.append(data[["Issues", "gap_flag"]])

    # Lags
    for k in LAGS:
        feats.append(data["Aggregate"].shift(k).rename(f"lag_{k}").to_frame())
    feats.append(data["Aggregate"].shift(SEASONAL_LAG).rename(f"lag_{SEASONAL_LAG}").to_frame())

    # Rolling (shift 1 to avoid leakage)
    base = data["Aggregate"].shift(1)
    for w in ROLL_WINDOWS:
        feats.append(base.rolling(w).mean().rename(f"roll_mean_{w}").to_frame())
        feats.append(base.rolling(w).std().rename(f"roll_std_{w}").to_frame())
        feats.append(base.rolling(w).max().rename(f"roll_max_{w}").to_frame())

    Xdf = pd.concat(feats, axis=1)

    # Add house id feature
    Xdf["house_id"] = house_id

    # Targets: next H steps (multi-output)
    Y = np.stack([data["Aggregate"].shift(-h).to_numpy() for h in range(1, HORIZON_H + 1)], axis=1)

    # Align/drop NaNs from lags/rolling/horizon
    # Build Y as a DataFrame so dropna applies to ALL horizon steps
    Ydf = pd.DataFrame(
        Y,
        index=Xdf.index,
        columns=[f"y_{h}" for h in range(1, HORIZON_H + 1)]
    )

    full = pd.concat([Xdf, Ydf], axis=1).dropna()  # drops if ANY feature or ANY horizon is NaN

    X = full[Xdf.columns].to_numpy(dtype=np.float32)
    Y = full[Ydf.columns].to_numpy(dtype=np.float32)

    return house_id, Xdf.columns.tolist(), X, Y

def time_split(X, Y):
    n = len(X)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:], Y[n_train+n_val:]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def main():
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched {RAW_GLOB}. Put CSVs in data/raw/.")

    print("Found files:")
    for f in files:
        print("  -", f)

    Xtr_list, Ytr_list = [], []
    Xva_list, Yva_list = [], []
    Xte_list, Yte_list = [], []
    feature_names = None

    for fp in files:
        house_id, cols, X, Y = build_XY_for_house(fp)

        # ensure consistent feature order
        if feature_names is None:
            feature_names = cols
        else:
            if cols != feature_names:
                raise ValueError("Feature columns mismatch across houses. (Unexpected)")

        X_train, Y_train, X_val, Y_val, X_test, Y_test = time_split(X, Y)

        Xtr_list.append(X_train); Ytr_list.append(Y_train)
        Xva_list.append(X_val);   Yva_list.append(Y_val)
        Xte_list.append(X_test);  Yte_list.append(Y_test)

        print(f"House {house_id}: X={X.shape}, Train/Val/Test = {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")

    # Combine across houses
    X_train = np.vstack(Xtr_list); Y_train = np.vstack(Ytr_list)
    X_val   = np.vstack(Xva_list); Y_val   = np.vstack(Yva_list)
    X_test  = np.vstack(Xte_list); Y_test  = np.vstack(Yte_list)

    # Save
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "Y_train.npy"), Y_train)
    np.save(os.path.join(OUT_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(OUT_DIR, "Y_val.npy"),   Y_val)
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUT_DIR, "Y_test.npy"),  Y_test)

    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        for c in feature_names:
            f.write(c + "\n")

    print("\nDone preprocessing MULTI-HOUSE.")
    print("Saved arrays to", OUT_DIR)
    print("Shapes:")
    print("  X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("  X_val:  ", X_val.shape,   "Y_val:  ", Y_val.shape)
    print("  X_test: ", X_test.shape,  "Y_test: ", Y_test.shape)

if __name__ == "__main__":
    main()