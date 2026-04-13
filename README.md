# Home Energy Optimization (INSE 6450 — Final Project)

This repository implements a complete AI system for **home energy optimization** using short-horizon load forecasting on the REFIT dataset. It covers:

- **Forecasting (Milestone 2):** multi-step (H=12) baseline model with engineered features
- **Robustness + Security + Monitoring (Milestone 3):** stress tests, FGSM, conformal uncertainty, drift dashboards
- **Continual Learning + HITL (Milestone 4):** replay-based micro-updates and active learning simulation

---

## Goal

Use smart-home electricity data to:
1) forecast short-term household energy demand, and  
2) support simple optimization ideas (peak warnings / low-load window suggestions).

---

## Dataset

**REFIT: Electrical Load Measurements (Cleaned)** — Zenodo (CC BY 4.0)  
Link: https://zenodo.org/records/5063428

Raw CSVs used (multi-house):
- `CLEAN_House1.csv`
- `CLEAN_House2.csv`
- `CLEAN_House3.csv`
- `CLEAN_House4.csv`
- `CLEAN_House5.csv`
- `CLEAN_House11.csv`

Key fields:
- `Unix` (timestamp in seconds)
- `Aggregate` (W) — target
- `Issues` — data-quality flag

---

## Task

Multi-step forecasting (regression):
- Resample to **5-minute** intervals
- Forecast horizon: **H = 12** steps = **1 hour ahead**
- Input: 24 engineered features
- Output: next 12 aggregate-load values

---

## Repository Structure

```
project-root/
  data/
    raw/                  # REFIT CSVs (not committed if large)
    processed/             # generated .npy arrays + feature_names.txt
  outputs/
    linear_forecaster.pt
    robust_scaler.joblib
    learning_curve_loss.png
    learning_curve_mae.png
    milestone3/
      stress_noise.csv
      stress_mask.csv
      fgsm.csv
      conformal_coverage.csv
      latency.csv
      robustness_curve_fgsm.png
      calibration_conformal.png
      monitoring_dashboard_psi.png
      monitoring_dashboard_js.png
      adaptation_results.txt
      resolved_cases.csv
      linear_forecaster_adapted.pt
    milestone4/
      continual_metrics.csv
      continual_updates.csv
      continual_trajectory_mae.png
      active_learning_metrics.csv
      active_learning_curve.png
      linear_forecaster_continual.pt
      linear_forecaster_active_learning.pt
  figures/                 # figures used in the final report (optional)
  src/
    preprocess_refit.py
    train_eval_linear.py
    milestone3_robustness.py
    milestone3_monitoring.py
    milestone3_adaptation.py
    milestone4_continual.py
    milestone4_active_learning.py
  requirements.txt
  references.bib           # for report (optional)
  main.tex                 # ICML-format final report (optional)
```

---

## Setup

### 1) Create folders (project root)

**Mac/Linux**
```bash
mkdir -p data/raw data/processed outputs
```

**Windows (PowerShell)**
```powershell
mkdir data\raw, data\processed, outputs
```

### 2) Create and activate a virtual environment (recommended)

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Mac/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick check:
```bash
python -c "import torch, sklearn, pandas, numpy; print('Imports OK'); print('Torch:', torch.__version__)"
```

---

## Place Raw Data

Put cleaned REFIT CSVs into:

```
data/raw/CLEAN_House1.csv
data/raw/CLEAN_House2.csv
data/raw/CLEAN_House3.csv
data/raw/CLEAN_House4.csv
data/raw/CLEAN_House5.csv
data/raw/CLEAN_House11.csv
```

---

## Milestone 2 — Baseline Forecasting Pipeline

### 1) Preprocess (creates processed arrays)
```bash
python src/preprocess_refit.py
```

Expected outputs in `data/processed/`:
- `X_train.npy`, `Y_train.npy`
- `X_val.npy`, `Y_val.npy`
- `X_test.npy`, `Y_test.npy`
- `feature_names.txt`

Optional sanity check:
```bash
python -c "import numpy as np; Y=np.load('data/processed/Y_test.npy'); print('NaNs in Y_test:', np.isnan(Y).sum())"
```

### 2) Train + evaluate baseline model
```bash
python src/train_eval_linear.py
```

Expected outputs in `outputs/`:
- `linear_forecaster.pt`
- `robust_scaler.joblib`
- `learning_curve_loss.png`
- `learning_curve_mae.png`

---

## Milestone 3 — Robustness, Security, Monitoring, Adaptation

### 1) Robustness + stress tests + FGSM + conformal calibration
```bash
python src/milestone3_robustness.py
```

Outputs in `outputs/milestone3/`:
- Stress tests: `stress_noise.csv`, `stress_mask.csv`
- Adversarial: `fgsm.csv`, `robustness_curve_fgsm.png`
- Uncertainty: `conformal_coverage.csv`, `calibration_conformal.png`
- Efficiency: `latency.csv`

### 2) Monitoring dashboards (PSI / JS)
```bash
python src/milestone3_monitoring.py
```

Outputs:
- `monitoring_dashboard_psi.png`
- `monitoring_dashboard_js.png`

### 3) Adaptation experiment (drift → update → improvement)
```bash
python src/milestone3_adaptation.py
```

Outputs:
- `adaptation_results.txt`
- `resolved_cases.csv`
- `linear_forecaster_adapted.pt`

---

## Milestone 4 — Continual Learning + HITL / Active Learning

### 1) Continual learning under simulated drift
```bash
python src/milestone4_continual.py
```

Outputs in `outputs/milestone4/`:
- `continual_metrics.csv`
- `continual_updates.csv`
- `continual_trajectory_mae.png`
- `linear_forecaster_continual.pt`

### 2) Active learning / HITL simulation
```bash
python src/milestone4_active_learning.py
```

Outputs:
- `active_learning_metrics.csv`
- `active_learning_curve.png`
- `linear_forecaster_active_learning.pt`

---

## Key Takeaways

- Baseline model is extremely lightweight (**300 parameters**) and fast (CPU inference ~0.012 ms/sample).
- Robustness:
  - stable under mild random noise,
  - degrades when key lag features are removed,
  - FGSM increases error monotonically.
- Conformal intervals provide conservative uncertainty estimates.
- Drift dashboards highlight shifting distributions; per-house monitoring is recommended.
- Continual learning and HITL loops are low-cost and demonstrate measurable improvements.

---

## Citation / License

REFIT dataset is CC BY 4.0.  
If you use this repo, cite: https://zenodo.org/records/5063428
