import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# Paths / output
# ---------------------------
DATA_DIR = "data/processed"
OUT_DIR = "outputs/milestone4"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = "outputs/linear_forecaster.pt"
SCALER_PATH = "outputs/robust_scaler.joblib"
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, "feature_names.txt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Experiment knobs
# ---------------------------
HUBER_DELTA = 1.0
MAPE_THRESH_W = 50.0

WINDOW = 20000            # evaluate each streaming window
DRIFT_STRENGTH = 0.10     # synthetic drift strength in scaled feature space

# Update trigger: start updating after drift begins, every other window
UPDATE_EVERY_K_WINDOWS = 2

# Continual update settings
UPDATE_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
PROX_LAMBDA = 1e-3        # penalty to previous weights

RECENT_BATCH = 30000      # how much recent labeled data to use
REPLAY_BATCH = 30000      # how much replay data to use

BUFFER_SIZE = 100000      # replay buffer capacity
BATCH_SIZE = 512          # SGD minibatch size for updates

RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)


# ---------------------------
# Helpers
# ---------------------------
def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Metrics flattened across horizon."""
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))

    # thresholded MAPE to avoid near-zero explosion
    m = np.abs(yt) >= MAPE_THRESH_W
    mape = (np.mean(np.abs((yt[m] - yp[m]) / np.abs(yt[m]))) * 100.0) if np.any(m) else float("nan")
    return mae, rmse, mape


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        pred = model(xb).cpu().numpy()
    return pred


def latency_ms(model: nn.Module, X: np.ndarray, n=500) -> tuple[float, float]:
    """Returns (p50_ms, p90_ms) for batch=1 inference."""
    model.eval()
    # warmup
    xb = torch.tensor(X[:1], dtype=torch.float32, device=DEVICE)
    for _ in range(50):
        _ = model(xb)

    times = []
    n = min(n, len(X))
    for i in range(n):
        xb = torch.tensor(X[i:i+1], dtype=torch.float32, device=DEVICE)
        t0 = time.perf_counter()
        _ = model(xb)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times = np.array(times)
    return float(np.percentile(times, 50)), float(np.percentile(times, 90))


class ReplayBuffer:
    """Reservoir sampling replay buffer."""
    def __init__(self, max_size: int, x_dim: int, y_dim: int):
        self.max_size = max_size
        self.X = np.zeros((max_size, x_dim), dtype=np.float32)
        self.Y = np.zeros((max_size, y_dim), dtype=np.float32)
        self.size = 0
        self.n_seen = 0

    def add_batch(self, Xb: np.ndarray, Yb: np.ndarray):
        for i in range(len(Xb)):
            self.n_seen += 1
            if self.size < self.max_size:
                self.X[self.size] = Xb[i]
                self.Y[self.size] = Yb[i]
                self.size += 1
            else:
                j = np.random.randint(0, self.n_seen)
                if j < self.max_size:
                    self.X[j] = Xb[i]
                    self.Y[j] = Yb[i]

    def sample(self, n: int):
        n = min(n, self.size)
        idx = np.random.choice(self.size, size=n, replace=False)
        return self.X[idx], self.Y[idx]


def simulate_drift(X_scaled: np.ndarray, feature_names: list[str], strength: float) -> np.ndarray:
    """
    Synthetic drift applied in scaled feature space:
      - add a small bias + scaling on continuous features,
      - do NOT perturb categorical/binary features (house_id, Issues, gap_flag, is_weekend).
    """
    Xd = X_scaled.copy()

    protected = set(["house_id", "Issues", "gap_flag", "is_weekend"])
    mask = np.ones(Xd.shape[1], dtype=np.float32)
    for i, n in enumerate(feature_names):
        if n in protected:
            mask[i] = 0.0

    # Apply drift only where mask=1
    Xd = Xd + strength * mask
    Xd = Xd * (1.0 + strength * mask)
    return Xd


def continual_update(model: nn.Module,
                     X_recent: np.ndarray, Y_recent: np.ndarray,
                     X_replay: np.ndarray, Y_replay: np.ndarray) -> float:
    """One continual learning update step. Returns update time in seconds."""
    model.train()
    X = np.vstack([X_recent, X_replay]).astype(np.float32)
    Y = np.vstack([Y_recent, Y_replay]).astype(np.float32)

    ds = TensorDataset(torch.tensor(X), torch.tensor(Y))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    # Snapshot old weights (prox regularization)
    old_state = {k: v.detach().clone().to(DEVICE) for k, v in model.state_dict().items()}

    t0 = time.time()
    for _ in range(UPDATE_EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)

            # proximal penalty
            prox = 0.0
            for k, v in model.state_dict().items():
                prox = prox + torch.sum((v - old_state[k]) ** 2)
            loss = loss + PROX_LAMBDA * prox

            loss.backward()
            opt.step()

    return time.time() - t0


def main():
    # ---------------------------
    # Load data
    # ---------------------------
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = [x.strip() for x in f.readlines()]

    scaler = load(SCALER_PATH)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    input_dim = X_train_s.shape[1]
    H = Y_train.shape[1]

    # ---------------------------
    # Load base model
    # ---------------------------
    model = nn.Linear(input_dim, H).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    # ---------------------------
    # Initialize replay buffer
    # ---------------------------
    buf = ReplayBuffer(BUFFER_SIZE, input_dim, H)
    # seed with training subset
    seed_n = min(BUFFER_SIZE, len(X_train_s))
    buf.add_batch(X_train_s[:seed_n], Y_train[:seed_n])

    # ---------------------------
    # Build streaming test with drift in 2nd half
    # ---------------------------
    X_stream = X_test_s.copy()
    drift_start = len(X_stream) // 2
    X_stream[drift_start:] = simulate_drift(X_stream[drift_start:], feature_names, DRIFT_STRENGTH)

    # ---------------------------
    # Continual loop
    # ---------------------------
    metrics_rows = []
    update_rows = []

    # baseline inference latency (clean)
    p50_ms, p90_ms = latency_ms(model, X_test_s, n=500)

    win_id = 0
    for start in range(0, len(X_stream) - WINDOW, WINDOW):
        end = start + WINDOW
        Xw = X_stream[start:end]
        Yw = Y_test[start:end]

        # Before update
        pred = predict(model, Xw)
        mae, rmse, mape = safe_metrics(Yw, pred)
        metrics_rows.append([win_id, start, end, "before", mae, rmse, mape])

        # Decide if update
        do_update = (start >= drift_start) and (win_id % UPDATE_EVERY_K_WINDOWS == 0)

        if do_update:
            # Recent batch (simulate new labels available)
            r0 = max(0, end - RECENT_BATCH)
            X_recent = X_stream[r0:end]
            Y_recent = Y_test[r0:end]

            X_rep, Y_rep = buf.sample(REPLAY_BATCH)
            upd_time = continual_update(model, X_recent, Y_recent, X_rep, Y_rep)

            # Add recent labeled to buffer
            buf.add_batch(X_recent, Y_recent)

            # After update (same window)
            pred2 = predict(model, Xw)
            mae2, rmse2, mape2 = safe_metrics(Yw, pred2)
            metrics_rows.append([win_id, start, end, "after", mae2, rmse2, mape2])

            update_rows.append([win_id, start, upd_time, buf.size])

        win_id += 1

    # ---------------------------
    # Save outputs
    # ---------------------------
    metrics_rows = np.array(metrics_rows, dtype=object)
    update_rows = np.array(update_rows, dtype=object)

    metrics_path = os.path.join(OUT_DIR, "continual_metrics.csv")
    updates_path = os.path.join(OUT_DIR, "continual_updates.csv")
    np.savetxt(metrics_path, metrics_rows, delimiter=",",
               header="window_id,win_start,win_end,phase,mae,rmse,mape_ge50",
               comments="", fmt="%s")
    np.savetxt(updates_path, update_rows, delimiter=",",
               header="window_id,win_start,update_time_sec,buffer_size",
               comments="", fmt="%s")

    # Plot MAE trajectory
    before = metrics_rows[metrics_rows[:, 3] == "before"]
    after  = metrics_rows[metrics_rows[:, 3] == "after"]

    plt.figure()
    plt.plot(before[:, 0].astype(int), before[:, 4].astype(float), marker="o", label="MAE before update")
    if len(after) > 0:
        plt.plot(after[:, 0].astype(int), after[:, 4].astype(float), marker="o", label="MAE after update")
    plt.axvline((drift_start // WINDOW), linestyle="--", label="drift start (approx)")
    plt.xlabel("Window id")
    plt.ylabel("MAE (W)")
    plt.title("Continual learning under simulated drift (MAE trajectory)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "continual_trajectory_mae.png"), dpi=200)

    # Save updated model snapshot
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "linear_forecaster_continual.pt"))

    # Print summary
    print("Saved:")
    print(" -", metrics_path)
    print(" -", updates_path)
    print(" -", os.path.join(OUT_DIR, "continual_trajectory_mae.png"))
    print(f"Inference latency (clean): p50={p50_ms:.4f} ms, p90={p90_ms:.4f} ms on {DEVICE}")

if __name__ == "__main__":
    main()