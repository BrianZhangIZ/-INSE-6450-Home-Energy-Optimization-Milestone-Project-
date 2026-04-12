import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import load, dump
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = "data/processed"
OUT_DIR = "outputs/milestone3"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HUBER_DELTA = 1.0
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 10  # fast adaptation
BATCH = 512

def metrics(y_true, y_pred):
    yt = y_true.reshape(-1); yp = y_pred.reshape(-1)
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return mae, rmse

def predict(model, X):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return model(xb).cpu().numpy()

def train_quick(model, X, Y):
    model.train()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    model.to(DEVICE)
    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model

def main():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    scaler = load("outputs/robust_scaler.joblib")
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Load original model
    input_dim = X_train_s.shape[1]; H = Y_train.shape[1]
    base = nn.Linear(input_dim, H).to(DEVICE)
    base.load_state_dict(torch.load("outputs/linear_forecaster.pt", map_location="cpu"))

    # --- Simulate drift on TEST (example: bias + scaling on continuous features) ---
    with open(os.path.join(DATA_DIR, "feature_names.txt"), "r") as f:
        names = [x.strip() for x in f.readlines()]
    idx_house = names.index("house_id") if "house_id" in names else None
    idx_binary = [i for i,n in enumerate(names) if n in ["Issues","gap_flag","is_weekend"]]

    mask = np.ones(input_dim, dtype=np.float32)
    if idx_house is not None: mask[idx_house] = 0.0
    for i in idx_binary: mask[i] = 0.0

    X_drift = X_test_s.copy()
    X_drift = X_drift + (0.15 * mask)  # add small bias in scaled space
    X_drift = X_drift * (1.10 * mask + (1.0 - mask))  # scale only continuous

    # Evaluate base model on drift
    pred_before = predict(base, X_drift)
    mae_before, rmse_before = metrics(Y_test, pred_before)

    # --- Adaptation: retrain quickly on a drifted TRAIN set ---
    X_train_d = X_train_s.copy()
    X_train_d = X_train_d + (0.15 * mask)
    X_train_d = X_train_d * (1.10 * mask + (1.0 - mask))

    adapted = nn.Linear(input_dim, H).to(DEVICE)
    adapted.load_state_dict(base.state_dict())
    adapted = train_quick(adapted, X_train_d[:200000], Y_train[:200000])  # subset for speed

    pred_after = predict(adapted, X_drift)
    mae_after, rmse_after = metrics(Y_test, pred_after)

    with open(os.path.join(OUT_DIR, "adaptation_results.txt"), "w") as f:
        f.write(f"Before adaptation (drifted test): MAE={mae_before:.2f}, RMSE={rmse_before:.2f}\n")
        f.write(f"After  adaptation (drifted test): MAE={mae_after:.2f}, RMSE={rmse_after:.2f}\n")

    # --- “Resolved cases”: pick top errors before, show improvement after ---
    err_before = np.mean(np.abs(Y_test - pred_before), axis=1)
    top_idx = np.argsort(-err_before)[:10]
    rows = []
    for i in top_idx:
        rows.append([
            int(i),
            float(err_before[i]),
            float(np.mean(np.abs(Y_test[i] - pred_after[i]))),
            float(np.mean(Y_test[i])),
            float(np.mean(pred_before[i])),
            float(np.mean(pred_after[i])),
        ])
    np.savetxt(os.path.join(OUT_DIR, "resolved_cases.csv"), np.array(rows), delimiter=",",
               header="index,mae_before,mae_after,true_mean,pred_before_mean,pred_after_mean", comments="")

    torch.save(adapted.state_dict(), os.path.join(OUT_DIR, "linear_forecaster_adapted.pt"))

    print("Saved adaptation results to outputs/milestone3/")

if __name__ == "__main__":
    main()