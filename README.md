# Home Energy Optimization (INSE 6450 — Milestone Project)

## Goal
Use smart-home electricity data to:
1) forecast short-term household energy demand, and
2) support simple energy optimization ideas (e.g., identify high-usage periods and shift flexible appliance usage).

## Dataset
REFIT: Electrical Load Measurements (Cleaned) — Zenodo (CC BY 4.0)  
Link: https://zenodo.org/records/5063428

Files used (Milestone 2):
- `CLEAN_House1.csv`
- `CLEAN_House2.csv`
- `CLEAN_House3.csv`
- `CLEAN_House4.csv`
- `CLEAN_House5.csv`
- `CLEAN_House11.csv`

Key fields used in the pipeline:
- `Unix` (timestamp in seconds) → converted to datetime
- `Aggregate` (W) (main target)
- `Issues` (binary flag for potential data-quality issues)

## Current Progress (Milestone 2)
### Task
Multi-step time-series forecasting (regression):
- Predict the next **12 steps** of aggregate load (H=12) after resampling.
- With 5-minute resampling, this corresponds to a **1-hour forecast horizon**.

### Preprocessing / Feature Engineering
- Resample raw readings to a fixed interval (**5 minutes**) to handle irregular timestamps.
- Build a supervised dataset with engineered features (24 total), including:
  - Time encodings: hour/day cyclical features + weekend flag
  - Lag features: recent lags + 24-hour seasonal lag (lag_288)
  - Rolling statistics over 1h / 6h / 24h windows (mean/std/max)
  - Data quality signals: `Issues` + `gap_flag`
  - `house_id` feature for multi-house training
- Temporal split per house (no leakage):
  - Train = 70% earliest
  - Val = next 15%
  - Test = final 15%
- Saved arrays in `data/processed/`:
  - `X_train.npy`, `Y_train.npy`, `X_val.npy`, `Y_val.npy`, `X_test.npy`, `Y_test.npy`
  - `feature_names.txt`

### Model (Milestone 2)
- Model: PyTorch `nn.Linear` multi-output forecaster (input_dim=24, output_dim=12)
- Loss: Huber loss (robust to heavy-tailed spikes)
- Scaling: RobustScaler (fit on train only)
- Metrics reported:
  - MAE / RMSE / MAPE (MAPE computed only when true load ≥ 50 W)
  - Peak-slice metrics (Top 5% true load)

## Repo Structure
- `data/raw/`         raw REFIT CSVs (not committed if large)
- `data/processed/`   generated `.npy` arrays + feature list
- `outputs/`          trained model + scaler + plots
- `src/`
  - `preprocess_refit.py`      multi-house preprocessing + feature generation + splits
  - `train_eval_linear.py`     train + evaluate linear model + save learning curves
- `requirements.txt`

## How to Run
```bash
pip install -r requirements.txt
python src/preprocess_refit.py
python src/train_eval_linear.py
