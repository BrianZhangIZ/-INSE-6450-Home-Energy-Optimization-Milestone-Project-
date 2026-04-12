import os, time
import numpy as np
import torch
from torch import nn
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = "data/processed"
OUT_DIR = "outputs/milestone3"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HUBER_DELTA = 1.0
MAPE_THRESH_W = 50.0
ALPHA_LIST = [0.2, 0.1, 0.05]  # 80/90/95% conformal intervals
FGSM_EPS = [0.0, 0.01, 0.05, 0.10]
NOISE_SIGMA = [0.0, 0.01, 0.05, 0.10]  # in scaled feature space

def load_feature_names():
    with open(os.path.join(DATA_DIR, "feature_names.txt"), "r") as f:
        return [x.strip() for x in f.readlines()]

def safe_metrics(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    m = np.abs(yt) >= MAPE_THRESH_W
    mape = (np.mean(np.abs((yt[m]-yp[m]) / np.abs(yt[m]))) * 100.0) if np.any(m) else float("nan")
    return mae, rmse, mape

def peak_slice(y_true, y_pred, top_pct=5):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    thresh = np.percentile(yt, 100-top_pct)
    mask = yt >= thresh
    mae = mean_absolute_error(yt[mask], yp[mask])
    rmse = np.sqrt(mean_squared_error(yt[mask], yp[mask]))
    return mae, rmse, thresh

def latency_stats(model, X, n=500):
    # returns p50/p90 ms and throughput
    model.eval()
    x = torch.tensor(X[:1], dtype=torch.float32, device=DEVICE)
    # warmup
    for _ in range(50):
        _ = model(x)
    times = []
    for i in range(min(n, len(X))):
        xb = torch.tensor(X[i:i+1], dtype=torch.float32, device=DEVICE)
        t0 = time.perf_counter()
        _ = model(xb)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000.0)
    times = np.array(times)
    p50 = float(np.percentile(times, 50))
    p90 = float(np.percentile(times, 90))
    throughput = 1000.0 / p50 if p50 > 0 else float("inf")
    return p50, p90, throughput

def build_model(input_dim, H, state_path):
    model = nn.Linear(input_dim, H)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    return model.to(DEVICE)

def predict(model, X):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        pred = model(xb).cpu().numpy()
    return pred

def fgsm_attack(model, X, Y, eps, feature_mask):
    # feature_mask: 1 for features allowed to perturb, 0 otherwise
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, device=DEVICE, requires_grad=True)
    y = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    grad = x.grad.detach()
    mask = torch.tensor(feature_mask, dtype=torch.float32, device=DEVICE).view(1, -1)
    adv = x + eps * torch.sign(grad) * mask
    return adv.detach().cpu().numpy()

def conformal_quantiles(y_true, y_pred, alpha):
    # residuals per horizon dimension (H)
    resid = np.abs(y_true - y_pred)  # (N,H)
    n = resid.shape[0]
    q = np.quantile(resid, np.ceil((n+1)*(1-alpha))/n, axis=0, method="higher")
    return q  # (H,)

def coverage(y_true, y_pred, q):
    lo = y_pred - q
    hi = y_pred + q
    inside = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(inside)), float(np.mean(hi - lo))

def main():
    # Load arrays
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    Y_val   = np.load(os.path.join(DATA_DIR, "Y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    # Load scaler + scale
    scaler = load("outputs/robust_scaler.joblib")
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    feature_names = load_feature_names()
    input_dim = X_train_s.shape[1]
    H = Y_train.shape[1]

    # Build model
    model = build_model(input_dim, H, "outputs/linear_forecaster.pt")

    # Identify feature indices
    idx_house = feature_names.index("house_id") if "house_id" in feature_names else None
    idx_binary = [i for i,n in enumerate(feature_names) if n in ["Issues","gap_flag","is_weekend"]]
    # allow perturbation on continuous lag/rolling/time features but NOT house_id or binary flags
    feature_mask = np.ones(input_dim, dtype=np.float32)
    if idx_house is not None:
        feature_mask[idx_house] = 0.0
    for i in idx_binary:
        feature_mask[i] = 0.0

    # --- Clean metrics ---
    pred_clean = predict(model, X_test_s)
    mae, rmse, mape = safe_metrics(Y_test, pred_clean)
    pmae, prmse, pth = peak_slice(Y_test, pred_clean, top_pct=5)

    print("\nCLEAN TEST:")
    print(f"MAE={mae:.2f} RMSE={rmse:.2f} MAPE(>=50W)={mape:.2f}% | Peak@Top5% thr={pth:.2f}W MAE={pmae:.2f} RMSE={prmse:.2f}")

    # --- Stress tests: noise ---
    noise_rows = []
    for s in NOISE_SIGMA:
        Xn = X_test_s.copy()
        if s > 0:
            # noise proportional to per-feature std (scaled space -> std ~ 1-ish, but compute anyway)
            std = X_test_s.std(axis=0, keepdims=True) + 1e-8
            noise = np.random.normal(0, s, size=Xn.shape) * std
            Xn = Xn + noise * feature_mask  # only perturb allowed features
        pred = predict(model, Xn)
        mae, rmse, mape = safe_metrics(Y_test, pred)
        noise_rows.append((s, mae, rmse, mape))
    np.savetxt(os.path.join(OUT_DIR, "stress_noise.csv"), np.array(noise_rows), delimiter=",",
               header="sigma,mae,rmse,mape_ge50", comments="")

    # --- Stress tests: feature masking ---
    def mask_features(X, names_to_mask):
        Xm = X.copy()
        for nm in names_to_mask:
            if nm in feature_names:
                Xm[:, feature_names.index(nm)] = 0.0
        return Xm

    mask_cases = {
        "mask_lag_288": ["lag_288"],
        "mask_all_roll": [n for n in feature_names if n.startswith("roll_")],
        "mask_some_lags": [n for n in feature_names if n.startswith("lag_") and n not in ["lag_1","lag_2"]],
    }
    mask_rows = []
    for cname, feats in mask_cases.items():
        Xm = mask_features(X_test_s, feats)
        pred = predict(model, Xm)
        mae, rmse, mape = safe_metrics(Y_test, pred)
        mask_rows.append((cname, mae, rmse, mape))
    with open(os.path.join(OUT_DIR, "stress_mask.csv"), "w") as f:
        f.write("case,mae,rmse,mape_ge50\n")
        for r in mask_rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f}\n")

    # --- OOD slice: per-house ---
    if idx_house is not None:
        ood_rows = []
        houses = np.unique(X_test[:, idx_house]).astype(int)
        for h in houses:
            m = X_test[:, idx_house] == h
            if m.sum() < 1000:
                continue
            pred = predict(model, X_test_s[m])
            mae, rmse, mape = safe_metrics(Y_test[m], pred)
            ood_rows.append((h, m.sum(), mae, rmse, mape))
        np.savetxt(os.path.join(OUT_DIR, "slice_by_house.csv"), np.array(ood_rows), delimiter=",",
                   header="house,n,mae,rmse,mape_ge50", comments="")

    # --- FGSM adversarial evaluation ---
    fgsm_rows = []
    for eps in FGSM_EPS:
        Xadv = fgsm_attack(model, X_test_s[:5000], Y_test[:5000], eps, feature_mask)  # subset for speed
        pred = predict(model, Xadv)
        mae, rmse, mape = safe_metrics(Y_test[:5000], pred)
        fgsm_rows.append((eps, mae, rmse, mape))
    np.savetxt(os.path.join(OUT_DIR, "fgsm.csv"), np.array(fgsm_rows), delimiter=",",
               header="eps,mae,rmse,mape_ge50", comments="")

    # Plot robustness curve
    plt.figure()
    plt.plot([r[0] for r in fgsm_rows], [r[1] for r in fgsm_rows], marker="o", label="MAE")
    plt.plot([r[0] for r in fgsm_rows], [r[2] for r in fgsm_rows], marker="o", label="RMSE")
    plt.xlabel("FGSM epsilon (scaled feature space)")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "robustness_curve_fgsm.png"), dpi=200)

    # --- Conformal intervals (calibration on validation) ---
    pred_val = predict(model, X_val_s)
    pred_test = pred_clean
    cover_rows = []
    for a in ALPHA_LIST:
        q = conformal_quantiles(Y_val, pred_val, alpha=a)  # (H,)
        cov, width = coverage(Y_test, pred_test, q)
        cover_rows.append((a, cov, width))
    np.savetxt(os.path.join(OUT_DIR, "conformal_coverage.csv"), np.array(cover_rows), delimiter=",",
               header="alpha,empirical_coverage,avg_interval_width", comments="")

    # Simple coverage plot (calibration-style)
    plt.figure()
    plt.plot([1-r[0] for r in cover_rows], [r[1] for r in cover_rows], marker="o")
    plt.plot([0.8,0.9,0.95],[0.8,0.9,0.95], linestyle="--")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "calibration_conformal.png"), dpi=200)

    # Latency stats (clean vs noise sigma=0.10)
    p50, p90, thr = latency_stats(model, X_test_s, n=500)
    Xn = X_test_s.copy()
    std = X_test_s.std(axis=0, keepdims=True) + 1e-8
    Xn = Xn + np.random.normal(0, 0.10, size=Xn.shape) * std * feature_mask
    p50n, p90n, thrn = latency_stats(model, Xn, n=500)
    with open(os.path.join(OUT_DIR, "latency.csv"), "w") as f:
        f.write("condition,p50_ms,p90_ms,throughput_samples_per_s\n")
        f.write(f"clean,{p50:.6f},{p90:.6f},{thr:.2f}\n")
        f.write(f"noise_sigma_0.10,{p50n:.6f},{p90n:.6f},{thrn:.2f}\n")

    print("\nSaved Milestone 3 robustness outputs to:", OUT_DIR)

if __name__ == "__main__":
    main()