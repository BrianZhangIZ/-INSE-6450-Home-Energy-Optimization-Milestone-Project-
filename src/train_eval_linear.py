import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from joblib import dump

DATA_DIR = "data/processed"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 1.0
EARLY_STOP_PATIENCE = 5

class NpyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

def metrics(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    # filter finite values (safety)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))

    # thresholded MAPE: avoid near-zero true values exploding the percentage
    thresh = 50.0  # watts (you can also try 100.0)
    m = np.abs(yt) >= thresh
    if np.any(m):
        mape = np.mean(np.abs((yt[m] - yp[m]) / np.abs(yt[m]))) * 100.0
    else:
        mape = float("nan")

    return mae, rmse, mape

def peak_slice_metrics(y_true, y_pred, top_pct=5):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    thresh = np.percentile(yt, 100 - top_pct)
    mask = yt >= thresh
    mae = mean_absolute_error(yt[mask], yp[mask])
    rmse = np.sqrt(mean_squared_error(yt[mask], yp[mask]))
    return mae, rmse, thresh

def eval_model(model, loader, loss_fn, device):
    model.eval()
    losses = []
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            losses.append(loss.item())
            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)
    return float(np.mean(losses)), y_true, y_pred

def main():
    # Load arrays
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    Y_val   = np.load(os.path.join(DATA_DIR, "Y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    # Scale features (fit on train only)
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    dump(scaler, os.path.join(OUT_DIR, "robust_scaler.joblib"))

    # Dataloaders
    train_loader = DataLoader(NpyDataset(X_train_s, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(NpyDataset(X_val_s,   Y_val),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(NpyDataset(X_test_s,  Y_test),  batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_train_s.shape[1]
    horizon_H = Y_train.shape[1]
    model = nn.Linear(input_dim, horizon_H).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    # Training loop + early stopping
    hist = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}
    best_val_mae = float("inf")
    best_state = None
    patience_left = EARLY_STOP_PATIENCE

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val_loss, yv_true, yv_pred = eval_model(model, val_loader, loss_fn, device)
        val_mae, val_rmse, val_mape = metrics(yv_true, yv_pred)

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_mae"].append(val_mae)
        hist["val_rmse"].append(val_rmse)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
              f"| val_MAE={val_mae:.2f} | val_RMSE={val_rmse:.2f} | val_MAPE={val_mape:.2f}%")

        # Early stopping on MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping triggered.")
                break

    total_train_time = time.time() - t0
    print(f"\nTotal training time: {total_train_time:.2f} seconds")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval: VAL + TEST
    _, yv_true, yv_pred = eval_model(model, val_loader, loss_fn, device)
    _, yt_true, yt_pred = eval_model(model, test_loader, loss_fn, device)

    val_mae, val_rmse, val_mape = metrics(yv_true, yv_pred)
    test_mae, test_rmse, test_mape = metrics(yt_true, yt_pred)

    peak_val_mae, peak_val_rmse, val_thresh = peak_slice_metrics(yv_true, yv_pred, top_pct=5)
    peak_test_mae, peak_test_rmse, test_thresh = peak_slice_metrics(yt_true, yt_pred, top_pct=5)

    print("\n=== Metrics (Flattened across horizon) ===")
    print(f"Validation: MAE={val_mae:.2f} | RMSE={val_rmse:.2f} | MAPE={val_mape:.2f}%")
    print(f"Test:       MAE={test_mae:.2f} | RMSE={test_rmse:.2f} | MAPE={test_mape:.2f}%")

    print("\n=== Peak Slice (Top 5% true load) ===")
    print(f"Val peak threshold ≈ {val_thresh:.2f} W | MAE={peak_val_mae:.2f} | RMSE={peak_val_rmse:.2f}")
    print(f"Test peak threshold ≈ {test_thresh:.2f} W | MAE={peak_test_mae:.2f} | RMSE={peak_test_rmse:.2f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "linear_forecaster.pt"))

    # Learning curve plots
    plt.figure()
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "learning_curve_loss.png"), dpi=200)

    plt.figure()
    plt.plot(hist["val_mae"], label="val MAE")
    plt.plot(hist["val_rmse"], label="val RMSE")
    plt.xlabel("epoch"); plt.ylabel("error"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "learning_curve_mae.png"), dpi=200)

    # Simple inference latency test (batch=1)
    model.eval()
    xb = torch.tensor(X_test_s[:1], dtype=torch.float32).to(device)
    # warmup
    for _ in range(20):
        _ = model(xb)
    t1 = time.time()
    for _ in range(200):
        _ = model(xb)
    latency_ms = (time.time() - t1) / 200 * 1000.0
    print(f"\nInference latency (batch=1): ~{latency_ms:.3f} ms/sample on {device}")

if __name__ == "__main__":
    main()