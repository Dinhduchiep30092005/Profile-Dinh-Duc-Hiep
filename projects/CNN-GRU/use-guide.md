# Hướng dẫn sử dụng — Bearing Fault Diagnosis CNN-GRU

## Yêu cầu môi trường

| Thư viện | Phiên bản |
|----------|-----------|
| Python | 3.10+ |
| PyTorch | 2.7.1+cu118 |
| NumPy | 2.4.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |

Kiểm tra GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Cấu trúc thư mục

```
Software/
├── Data/                          # Dữ liệu thô (đặt vào đây trước khi chạy)
│   ├── CWRU_Bearing_NumPy-main/Data/
│   ├── MFPT Fault Data Sets/
│   └── Paderborn/
│
├── processed_data/                # Sinh ra sau process_data.py (native rate)
├── processed_data_std/            # Sinh ra sau process_data.py --dataset cross_prep (12kHz)
├── Data_test/                     # Tín hiệu gốc đã split 70/15/15
├── saved_models/                  # Model và kết quả sau training
│
├── process_data.py                # Bước 1: xử lý dữ liệu thô
├── extract_data.py                # Kiến trúc FeatureExtractor (CNN branches)
├── train_model.py                 # Kiến trúc CNNGRU + hàm train()
├── train_cross.py                 # Train 9 cross-dataset scenarios
├── run.py                         # Runner gom nhiều job
├── inference.py                   # Inference / open-set classification
└── tools.py                       # Tiện ích kiểm tra checkpoint, inspect file
```

---

## Bước 1 — Xử lý dữ liệu

Chạy một lần duy nhất trước khi train. Đảm bảo thư mục `Data/` đã có đủ 3 dataset.

```bash
# Xử lý toàn bộ (khuyến nghị) — bao gồm luôn cross_prep, chạy lệnh này là đủ
python process_data.py --dataset all

# Hoặc xử lý từng dataset riêng lẻ
python process_data.py --dataset cwru
python process_data.py --dataset mfpt
python process_data.py --dataset paderborn
python process_data.py --dataset merged

# Nếu đã chạy từng dataset riêng (không dùng --dataset all),
# cần chạy thêm lệnh này để tạo processed_data_std/ cho train_cross.py
python process_data.py --dataset cross_prep
```

**Kết quả sinh ra:**
- `Data_test/{cwru,mfpt,paderborn,merged}/` — tín hiệu gốc đã split 70/15/15
- `processed_data/{cwru,mfpt,paderborn,merged}/` — windows 2048 điểm + z-score (native rate)
- `processed_data_std/{cwru,mfpt,paderborn}/` — windows @12kHz (dùng cho cross scenarios)

---

## Bước 2 — Train model

### 2a. Train từng dataset riêng lẻ (Individual)

```bash
# Train CWRU
python train_model.py --dataset cwru --epochs 200

# Train MFPT
python train_model.py --dataset mfpt --epochs 200

# Train Paderborn
python train_model.py --dataset paderborn --epochs 200

# Train Merged (mặc định nếu không truyền --dataset)
python train_model.py --epochs 200
```

Kết quả lưu tại `saved_models/{dataset}/`.

### 2b. Gom nhiều job bằng run.py

```bash
# Train cả 3 individual datasets (CWRU, MFPT, Paderborn)
python -u run.py --jobs individual --epochs 200

# Train merged
python -u run.py --jobs merged --epochs 200

# Train tất cả 4 jobs (individual + merged)
python -u run.py --jobs all --epochs 200

# Tiếp tục từ checkpoint nếu bị ngắt giữa chừng
python -u run.py --jobs individual --epochs 200 --resume
```

Dùng `-u` để stdout không bị buffer khi redirect vào file log:
```bash
python -u run.py --jobs individual --epochs 200 > train_log.txt
```

### 2c. Train 9 cross-dataset scenarios

Yêu cầu `processed_data_std/` đã được tạo (Bước 1 `--dataset cross_prep`).

```bash
# Train tất cả 9 scenarios (S1-S9)
python train_cross.py

# Chỉ train 1 scenario cụ thể
python train_cross.py --scenario S2

# Bỏ qua một số scenario
python train_cross.py --skip S1 S5 S9
```

Kết quả lưu tại `saved_models/S{N}_{train}_to_{test}/`.

### Các tùy chọn training

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--epochs` | Số epoch tối đa | 200 |
| `--lr` | Learning rate | 1e-3 |
| `--scheduler` | `plateau` / `cosine` / `onecycle` | plateau |
| `--resume` | Tiếp tục từ checkpoint | False |
| `--eta-min` | LR tối thiểu (cosine) | 1e-6 |

---

## Bước 3 — Kết quả sau training

Mỗi scenario tự động lưu vào `saved_models/{name}/`:

| File | Nội dung |
|------|---------|
| `cnn_gru_{name}.pth` | Model weights tốt nhất (theo val_loss) |
| `checkpoint_latest.pth` | Checkpoint epoch cuối (dùng để resume) |
| `training_results.json` | Toàn bộ history + metrics (loss, acc, std) |
| `training_curves.png` | Biểu đồ Loss và Accuracy theo epoch |
| `confusion_matrix.png` | Confusion Matrix trên tập test |
| `final_metrics.png` | Bar chart Train/Val/Test ± std |
| `lr_schedule.png` | Biểu đồ Learning Rate schedule |
| `classification_report.txt` | Precision / Recall / F1 per class |

---

## Bước 4 — Inference

Sử dụng model Merged đã train để phân loại tín hiệu mới.

```bash
# Chạy demo inference trên tập test của dataset merged
python inference.py

# Thay đổi threshold open-set (mặc định 0.7)
python inference.py --tau 0.8

# Inference trên dataset khác
python inference.py --dataset cwru --tau 0.7
```

**Cơ chế open-set:** nếu xác suất tối đa `p_max < τ` → dự đoán là `None` (lỗi không xác định), không cần train lại.

Dùng trong code:
```python
from inference import OpenSetClassifier
import numpy as np

clf = OpenSetClassifier(tau=0.7)

# Dự đoán 1 window đã z-score (2048 điểm)
window = np.load("my_signal.npy")   # shape (2048,)
result = clf.predict(window)
print(result['label'], result['confidence'])

# Dự đoán từ tín hiệu thô chưa cắt window
raw = np.load("raw_signal.npy")
result = clf.predict_from_raw(raw)
print(result['label'], result['votes'])
```

---

## Dashboard đánh giá model (GUI)

Giao diện đồ họa để test nhanh model đã train trên tín hiệu thô từ `Data_test/`, không cần viết code.

```bash
python dashboard.py
```

**Cách dùng:**
1. Chọn **Model** — 1 trong 4 model individual (`cwru`, `mfpt`, `paderborn`, `merged`) đã train ở Bước 2.
2. Chọn **Data test** — tín hiệu thô từ `Data_test/{dataset}/` (phần split test 15%, model chưa từng thấy).
3. Chỉnh **Threshold τ (open-set)** nếu cần (mặc định 0.7).
4. Bấm **TEST**.

**Cơ chế bên trong:**
- Tự động resample tín hiệu về đúng sample rate model đã train, nếu data test khác rate (vd. test model `merged` — train ở 12kHz — trên data `mfpt` — gốc 48,828Hz — sẽ tự resample 48,828→12kHz trước khi cắt window, tránh model nhận sai domain và đoán bậy).
- Pipeline: `tín hiệu thô → resample (nếu cần) → sliding_window(2048, stride 1024) → z-score per window → model predict → vote đa số → nhãn tín hiệu`.
- Nhãn ngoài class-space của model (vd. Ball khi test model MFPT/Paderborn, vốn chỉ có 3 lớp) sẽ tự động bị loại trước khi test, không gây lỗi.

**Kết quả hiển thị:**
- Accuracy, số tín hiệu bị vote "None" (từ chối vì open-set threshold), Classification Report theo tín hiệu.
- Confusion Matrix (chỉ hiện các nhãn thực sự xuất hiện trong kết quả, không hiện thừa nhãn model có nhưng data không có).
- Bảng chi tiết từng tín hiệu: file gốc, nhãn thật/dự đoán, confidence, số window, OK/ERR — panel có thanh trượt ngang + dọc để xem hết khi dòng dài.

---

## Tiện ích — tools.py

```bash
# Xem tiến độ tất cả model đã train (epoch, acc, loss, LR, thời gian)
python tools.py check

# Xem thông tin file dữ liệu .npy
python tools.py inspect processed_data/cwru/X_train.npy

# Xem thông tin model .pth (keys, layers, số params)
python tools.py inspect saved_models/cwru/cnn_gru_cwru.pth

# Tạo checkpoint để resume các cross jobs cũ
python tools.py make-checkpoints
```

---

## Kiểm tra module FeatureExtractor

Chạy để xác nhận kiến trúc CNN branches hoạt động đúng (không cần data):

```bash
python extract_data.py
```

Output mẫu:
```
PASSED - All shape and value checks OK!
```

---

## Thứ tự chạy đầy đủ từ đầu

```bash
# 1. Xử lý dữ liệu (--dataset all đã bao gồm cross_prep, chạy 1 lần là đủ)
python process_data.py --dataset all

# 2. Train 4 individual scenarios
python -u run.py --jobs individual --epochs 200
python -u run.py --jobs merged --epochs 200

# 3. Train 9 cross-dataset scenarios (processed_data_std/ đã có từ bước 1)
python train_cross.py

# 4. Kiểm tra kết quả
python tools.py check

# 5. Inference thử
python inference.py --dataset merged
```

---

## Không gian nhãn

| Nhãn | CWRU | MFPT | Paderborn |
|------|------|------|-----------|
| 0 | Normal | Baseline | Normal |
| 1 | Inner Race | Inner Race | Inner Race |
| 2 | Outer Race | Outer Race | Outer Race |
| 3 | Ball | — | — |

> **Lưu ý:** Nhãn 3 (Ball) chỉ có trong CWRU. Khi cross-dataset từ MFPT/Paderborn sang CWRU, các mẫu Ball bị lọc bỏ vì nằm ngoài không gian lớp của model.
