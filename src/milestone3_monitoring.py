import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

DATA_DIR = "data/processed"
OUT_DIR = "outputs/milestone3"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW = 5000  # streaming window size
PSI_BINS = 10

def psi(expected, actual, bins=10):
    # population stability index
    q = np.linspace(0, 1, bins+1)
    cuts = np.quantile(expected, q)
    cuts[0] = -np.inf; cuts[-1] = np.inf
    e = np.histogram(expected, bins=cuts)[0] / len(expected)
    a = np.histogram(actual, bins=cuts)[0] / len(actual)
    eps = 1e-8
    e = np.clip(e, eps, 1); a = np.clip(a, eps, 1)
    return float(np.sum((a - e) * np.log(a / e)))

def js_divergence(p, q):
    # p, q are histograms normalized
    eps = 1e-12
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    m = 0.5*(p+q)
    kl_pm = np.sum(p*np.log(p/m))
    kl_qm = np.sum(q*np.log(q/m))
    return float(0.5*(kl_pm + kl_qm))

def main():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    with open(os.path.join(DATA_DIR, "feature_names.txt"), "r") as f:
        names = [x.strip() for x in f.readlines()]

    scaler = load("outputs/robust_scaler.joblib")
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Choose key features to monitor
    key_feats = [n for n in ["lag_1", "lag_288", "roll_mean_12", "roll_max_288", "house_id"] if n in names]
    idx = [names.index(n) for n in key_feats]

    # Compute drift per window
    psi_scores = {n: [] for n in key_feats}
    js_scores = {n: [] for n in key_feats}
    x_axis = []

    # reference distributions from training
    ref = {n: X_train_s[:, names.index(n)] for n in key_feats}

    for start in range(0, len(X_test_s) - WINDOW, WINDOW):
        cur = X_test_s[start:start+WINDOW]
        x_axis.append(start)
        for n in key_feats:
            v_ref = ref[n]
            v_cur = cur[:, names.index(n)]

            psi_scores[n].append(psi(v_ref, v_cur, bins=PSI_BINS))

            # JS on histograms (same cuts)
            cuts = np.quantile(v_ref, np.linspace(0,1,PSI_BINS+1))
            cuts[0] = -np.inf; cuts[-1] = np.inf
            p = np.histogram(v_ref, bins=cuts)[0]; p = p / p.sum()
            q = np.histogram(v_cur, bins=cuts)[0]; q = q / q.sum()
            js_scores[n].append(js_divergence(p, q))

    # Dashboard plot (single figure with multiple lines)
    plt.figure(figsize=(10,5))
    for n in key_feats:
        plt.plot(x_axis, psi_scores[n], label=f"PSI {n}")
    plt.axhline(0.2, linestyle="--")
    plt.title("Drift Monitoring (PSI per window) — threshold ~0.2")
    plt.xlabel("stream offset (samples)")
    plt.ylabel("PSI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "monitoring_dashboard_psi.png"), dpi=200)

    plt.figure(figsize=(10,5))
    for n in key_feats:
        plt.plot(x_axis, js_scores[n], label=f"JS {n}")
    plt.title("Drift Monitoring (JS divergence per window)")
    plt.xlabel("stream offset (samples)")
    plt.ylabel("JS divergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "monitoring_dashboard_js.png"), dpi=200)

    print("Saved monitoring dashboards to outputs/milestone3/")

if __name__ == "__main__":
    main()