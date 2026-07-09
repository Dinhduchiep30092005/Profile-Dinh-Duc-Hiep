# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bearing fault diagnosis system using CNN-GRU deep learning on 3 benchmark vibration datasets: CWRU, MFPT, and Paderborn. Trains 13 scenarios total: 4 individual models + 9 cross-dataset generalization scenarios.

## Commands

### Data Processing
```bash
# Process all datasets (run this first, or after any data/rate changes)
python process_data.py --dataset all

# Process a single dataset
python process_data.py --dataset [cwru|mfpt|paderborn|merged|cross_prep]
# cross_prep = generates processed_data_std/ (12kHz) needed by train_cross.py
```

### Training
```bash
# Train individual models (cwru, mfpt, paderborn)
python -u run.py --jobs individual --epochs 200

# Train merged model
python -u run.py --jobs merged --epochs 200

# Train all 9 cross-dataset scenarios (S1–S9)
python train_cross.py

# Resume from checkpoint
python run.py --jobs individual --epochs 200 --resume
```

Use `-u` flag for unbuffered stdout when redirecting to log files.

### GPU Check
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Evaluation Dashboard (GUI)
```bash
python dashboard.py
```
Tkinter GUI to test a trained model against a `Data_test/{dataset}` raw-signal split without writing code (pick model, pick data, set open-set threshold τ, click TEST). Only lists the 4 individual models (`saved_models/{cwru,mfpt,paderborn,merged}/`), not the 9 cross scenarios.

### Inference (CLI)
```bash
python inference.py --dataset merged --tau 0.7
```
Runs `OpenSetClassifier` (in `inference.py`) against a dataset's test split for a quick open-set demo.

### Checkpoint/data inspection
```bash
python tools.py check                                       # training progress of all saved models
python tools.py inspect processed_data/cwru/X_train.npy      # inspect a .npy/.pth file
python tools.py make-checkpoints                             # backfill checkpoint_latest.pth for old cross jobs
```

## Architecture

### Data Pipeline

```
Raw Data/
  ├── CWRU_Bearing_NumPy-main/   (.npz, DE+FE channels, 12kHz)
  ├── MFPT Fault Data Sets/      (.mat, bearing.gs field, mixed 97,656/48,828Hz)
  └── Paderborn/                 (.mat, vibration_1 channel, 64kHz)
         ↓ process_data.py --dataset all
Data_test/                       ← raw signals split 70/15/15 at signal level
  ├── cwru/    @12kHz
  ├── mfpt/    @48,828Hz
  ├── paderborn/ @64kHz
  └── merged/  @12kHz (resampled when loading from above)

processed_data/                  ← windowed+z-scored, for individual training
  ├── cwru/    @12kHz  — 19k/4k/4k windows
  ├── mfpt/    @48,828Hz — 2.5k/569/569 windows
  ├── paderborn/ @64kHz — 31k/6.7k/6.7k windows
  └── merged/  @12kHz  — 25k/5.5k/5.2k windows

processed_data_std/              ← all @12kHz, for cross-dataset scenarios
  ├── cwru/    (same as processed_data/cwru)
  ├── mfpt/    (resampled 48,828→12kHz) — 616/142/142 windows
  └── paderborn/ (resampled 64k→12kHz) — 5.7k/1.2k/1.2k windows
```

**Critical design choice**: Split happens at **signal level** (before sliding window) to prevent data leakage. Windows from the same recording are never split across train/test sets.

### Model: CNNGRU (`train_model.py` + `extract_data.py`)

```
Input: (B, 2048) raw vibration window
    → FeatureExtractor (extract_data.py)
        Branch A — CNN1D temporal:
            InstanceNorm1d → Conv1d×3 (BN+GELU) → AdaptiveAvgPool1d(32) → (B, 32, 128)
        Branch B — CNN2D spectral:
            STFT(n_fft=1024, hop=256) → log power → Conv2d×3 (BN+GELU+MaxPool)
            → mean pool freq axis → AdaptiveAvgPool1d(32) → (B, 32, 128)
        Concat → (B, 32, 256)
    → GRU(input=256, hidden=128, layers=1, batch_first=True) → (B, 32, 128)
    → mean pool timesteps → (B, 128)
    → FC(128→64) + Dropout(0.3) + ReLU
    → FC(64→num_classes) → logits
```

### Scripts

| Script | Role |
|--------|------|
| `process_data.py` | Full data pipeline: read raw → resample → split signals → save to `Data_test/` → sliding window → z-score → save to `processed_data/` and `processed_data_std/` |
| `extract_data.py` | `FeatureExtractor` nn.Module (Branch A + Branch B). Imported by `train_model.py` and `train_cross.py` |
| `train_model.py` | `CNNGRU` model + `train()` function. Reads `processed_data/`. Saves to `saved_models/{dataset}/` |
| `run.py` | Runner for individual/merged scenarios. Calls `train_model.train()` |
| `train_cross.py` | Trains 9 cross-dataset scenarios (S1–S9). Reads `processed_data_std/`. Saves to `saved_models/S{N}_{train}_to_{test}/` |
| `dashboard.py` | Tkinter GUI: pick a saved model + a `Data_test/{dataset}` split, run open-set inference, view accuracy/confusion matrix/per-signal report |
| `inference.py` | `OpenSetClassifier` — 4-class softmax + threshold rule producing a 5th "None" (unknown fault) outcome |
| `tools.py` | CLI utilities: `check` (training progress), `inspect` (.npy/.pth contents), `make-checkpoints` |

### 13 Training Scenarios

**Individual (4)** — each dataset trains and tests on itself:
- CWRU, MFPT, Paderborn, Merged

**Cross (9)** — train on one dataset, test on another:
- S1–S3: Train CWRU → test CWRU/MFPT/Paderborn
- S4–S6: Train MFPT → test CWRU/MFPT/Paderborn
- S7–S9: Train Paderborn → test CWRU/MFPT/Paderborn

### Label Space

| Label | CWRU | MFPT | Paderborn |
|-------|------|------|-----------|
| 0 | Normal | Baseline | Normal (K001, K002, K006) |
| 1 | Inner Race | Inner Race | Inner Race (KI01, KI05, KI07) |
| 2 | Outer Race | Outer Race | Outer Race (KA01, KA05, KA07) |
| 3 | Ball | — | — |

Ball (label 3) exists only in CWRU. KB bearings (combined faults) are excluded from Paderborn.

### Open-Set Classification ("None" outcome)

The trained model (`CNNGRU`, `train_model.py`) always has exactly `num_classes` softmax outputs — it is never trained with a "None"/unknown label. The 5th outcome only exists at inference time (`inference.py`, `dashboard.py`): if `p_max = max(softmax(logits)) < τ` (default τ=0.7), the caller overrides the prediction to "None" by threshold rule, not by the model itself. This is a normal return value (`is_none: True` in the result dict), not an exception — downstream code decides how to handle it (flag for manual review, etc.).

### Frequency Standardization

- **Individual training**: native rates (CWRU: 12kHz, MFPT: 48,828Hz, Paderborn: 64kHz)
- **Merged + cross scenarios**: all resampled to 12kHz via `scipy.signal.resample_poly`
- MFPT baseline/OR-2 files are at 97,656Hz and get downsampled to 48,828Hz for within-MFPT consistency
- `Data_test/{dataset}/signals_*.npy` stores raw signals **pre-window, pre-z-score**, but already at whatever rate that dataset's model expects (native for cwru/mfpt/paderborn, 12kHz for merged) — z-score only happens later when windows are cut for `processed_data/`
- `dashboard.py` infers the sample rate a model expects from its folder name (`FS_BY_DATASET` map) and auto-resamples raw test signals to match before windowing — needed when testing one dataset's model against another dataset's raw signal (e.g. `merged` model, native-rate MFPT signal)

### Key Hyperparameters

```python
WINDOW_SIZE = 2048, STEP_SIZE = 1024  # 50% overlap
BATCH_SIZE = 64, LEARNING_RATE = 1e-3, NUM_EPOCHS = 200
GRU_HIDDEN_SIZE = 128, FC_HIDDEN_SIZE = 64, DROPOUT_RATE = 0.3
FEATURE_DIM = 128  # per branch; concat = 256
SEQUENCE_LEN = 32  # T — fixed output timesteps from both branches
```

Class weights: `w_k = sqrt(N / (K * n_k))` — sqrt inverse frequency (softer than linear, avoids loss instability for rare Ball class).

### Saved Model Outputs

Each scenario saves to `saved_models/{name}/`:
- `cnn_gru_{name}.pth` — best checkpoint (by val loss)
- `checkpoint_latest.pth` — latest epoch (for `--resume`)
- `training_results.json` — full history + metrics
- `training_curves.png`, `confusion_matrix.png`

### Paderborn-Specific Details

- Only operating condition `N15_M07_F10` (1500 rpm, 0.7 Nm, 1000 N) is used — other conditions excluded to avoid domain shift
- Each recording capped at 256,000 samples before windowing
- 20 files per bearing code × 9 bearings = 180 recordings total
