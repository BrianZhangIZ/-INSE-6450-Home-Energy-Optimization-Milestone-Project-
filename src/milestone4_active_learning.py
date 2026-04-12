import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = "data/processed"
OUT_DIR = "outputs/milestone4"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = "outputs/linear_forecaster.pt"
SCALER_PATH = "outputs/robust_scaler.joblib"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HUBER_DELTA = 1.0
MAPE_THRESH_W = 50.0

CYCLES = 5
QUERY_BUDGET = 5000        # number of samples "human labels" per cycle (simulation)
UPDATE_EPOCHS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512

RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))

    m = np.abs(yt) >= MAPE_THRESH_W
    mape = (np.mean(np.abs((yt[m] - yp[m]) / np.abs(yt[m]))) * 100.0) if np.any(m) else float("nan")
    return mae, rmse, mape

def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return model(xb).cpu().numpy()

def train_micro_update(model: nn.Module, X: np.ndarray, Y: np.ndarray) -> float:
    model.train()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    t0 = time.time()
    for _ in range(UPDATE_EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return time.time() - t0

def main():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    Y_val   = np.load(os.path.join(DATA_DIR, "Y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    scaler = load(SCALER_PATH)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    input_dim = X_train_s.shape[1]
    H = Y_train.shape[1]

    # Load base model
    base = nn.Linear(input_dim, H).to(DEVICE)
    base.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    # Start with base model weights
    model = nn.Linear(input_dim, H).to(DEVICE)
    model.load_state_dict(base.state_dict())

    # --- Build uncertainty proxy from validation residuals (conformal-like width) ---
    pred_val = predict(model, X_val_s)
    resid = np.abs(Y_val - pred_val)  # (N,H)
    q90 = np.quantile(resid, 0.90, axis=0)  # per-horizon residual quantile
    # per-sample uncertainty proxy: mean |pred| weighted by q90 (simple proxy)
    pred_test = predict(model, X_test_s)
    uncertainty = np.mean(np.abs(pred_test), axis=1) * float(np.mean(q90))

    # peak/impact proxy: higher predicted load = higher impact
    impact = np.mean(np.abs(pred_test), axis=1)

    # Hybrid score: uncertainty + impact (simple)
    score = 0.6 * uncertainty + 0.4 * impact

    # Active learning state
    labeled_idx = np.array([], dtype=int)

    rows = []
    # Initial evaluation
    pred0 = predict(model, X_test_s)
    mae0, rmse0, mape0 = safe_metrics(Y_test, pred0)
    rows.append([0, 0, mae0, rmse0, mape0, 0.0])

    for c in range(1, CYCLES + 1):
        # Query: pick top candidates by score, avoid repeats
        candidates = np.argsort(-score)
        candidates = candidates[~np.isin(candidates, labeled_idx)]

        k = min(QUERY_BUDGET, len(candidates))
        chosen = candidates[:k]
        labeled_idx = np.unique(np.concatenate([labeled_idx, chosen]))

        # "Human labels" are simulated by revealing Y_test
        X_lab = X_test_s[labeled_idx]
        Y_lab = Y_test[labeled_idx]

        # Mix labeled set with replay from training to reduce overfit to test
        rep_n = min(len(X_train_s), len(X_lab))
        rep_idx = np.random.choice(len(X_train_s), size=rep_n, replace=False)
        X_mix = np.vstack([X_lab, X_train_s[rep_idx]])
        Y_mix = np.vstack([Y_lab, Y_train[rep_idx]])

        update_time = train_micro_update(model, X_mix, Y_mix)

        pred = predict(model, X_test_s)
        mae, rmse, mape = safe_metrics(Y_test, pred)
        rows.append([c, len(X_lab), mae, rmse, mape, update_time])

    rows = np.array(rows, dtype=object)
    out_csv = os.path.join(OUT_DIR, "active_learning_metrics.csv")
    np.savetxt(out_csv, rows, delimiter=",",
               header="cycle,total_labeled,mae,rmse,mape_ge50,update_time_sec",
               comments="", fmt="%s")

    # Plot
    plt.figure()
    plt.plot(rows[:, 0].astype(int), rows[:, 2].astype(float), marker="o", label="MAE")
    plt.plot(rows[:, 0].astype(int), rows[:, 3].astype(float), marker="o", label="RMSE")
    plt.xlabel("Active learning cycle")
    plt.ylabel("Error")
    plt.title("Active learning / HITL simulation (error vs cycles)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "active_learning_curve.png"), dpi=200)

    # Save final updated model
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "linear_forecaster_active_learning.pt"))

    print("Saved:")
    print(" -", out_csv)
    print(" -", os.path.join(OUT_DIR, "active_learning_curve.png"))

if __name__ == "__main__":
    main()