# CNN-GRU — Bearing Fault Diagnosis (Chẩn đoán hư hỏng vòng bi bằng CNN-GRU)

> **Lưu ý về repo này:** Đây là bản *source code* của đồ án, được đẩy lên từ một dự án lớn hơn nhiều. Các thư mục dữ liệu thô/đã xử lý (`Data/`, `Data_test/`, `processed_data/`, `processed_data_std/`) **không được đưa lên GitHub** vì tổng dung lượng lên tới hơn **25GB**, vượt xa giới hạn của GitHub (100MB/file). File này mô tả lại toàn bộ dự án — bao gồm cả phần dữ liệu — để người xem hiểu được bức tranh đầy đủ dù không thấy các thư mục đó trong repo.

## 1. Giới thiệu

Đây là một hệ thống **chẩn đoán hư hỏng vòng bi (bearing fault diagnosis)** bằng deep learning, kết hợp kiến trúc **CNN 2 nhánh (1D + 2D) và GRU**. Mô hình được huấn luyện và đánh giá trên 3 bộ dữ liệu rung động chuẩn quốc tế:

- **CWRU** (Case Western Reserve University)
- **MFPT** (Machinery Failure Prevention Technology)
- **Paderborn** (Paderborn University)

Mục tiêu: phân loại tín hiệu rung động thành các loại hư hỏng (Normal, Inner Race, Outer Race, Ball) và đánh giá khả năng **tổng quát hóa chéo bộ dữ liệu** (cross-dataset generalization) — tức là mô hình huấn luyện trên bộ dữ liệu này có nhận diện tốt tín hiệu từ bộ dữ liệu khác hay không. Đây là bài toán thực tế quan trọng vì trong công nghiệp, mô hình thường phải hoạt động trên thiết bị/cảm biến khác với lúc huấn luyện.

Tổng cộng **13 kịch bản huấn luyện**: 4 mô hình riêng lẻ (individual) + 9 kịch bản cross-dataset (S1–S9).

## 2. Kiến trúc mô hình (CNN-GRU)

```
Input: (B, 2048) cửa sổ tín hiệu rung động thô
    → FeatureExtractor
        Nhánh A — CNN1D (miền thời gian):
            InstanceNorm1d → Conv1d×3 (BN+GELU) → AdaptiveAvgPool1d(32) → (B, 32, 128)
        Nhánh B — CNN2D (miền tần số):
            STFT(n_fft=1024, hop=256) → log power → Conv2d×3 (BN+GELU+MaxPool)
            → mean pool theo trục tần số → AdaptiveAvgPool1d(32) → (B, 32, 128)
        Ghép 2 nhánh → (B, 32, 256)
    → GRU(input=256, hidden=128, 1 lớp, batch_first=True) → (B, 32, 128)
    → mean pool theo thời gian → (B, 128)
    → FC(128→64) + Dropout(0.3) + ReLU
    → FC(64→num_classes) → logits
```

Ý tưởng chính: nhánh CNN1D bắt đặc trưng thời gian thô (biên độ, xung va đập), nhánh CNN2D-STFT bắt đặc trưng phổ tần số (tần số đặc trưng của từng loại hư hỏng), GRU tổng hợp thông tin theo chuỗi 32 bước thời gian trước khi phân loại.

**Siêu tham số chính:**
```
WINDOW_SIZE = 2048, STEP_SIZE = 1024 (overlap 50%)
BATCH_SIZE = 64, LEARNING_RATE = 1e-3, NUM_EPOCHS = 200
GRU_HIDDEN_SIZE = 128, FC_HIDDEN_SIZE = 64, DROPOUT_RATE = 0.3
```
Trọng số lớp (class weights): `w_k = sqrt(N / (K * n_k))` — giảm nhẹ mất cân bằng dữ liệu (đặc biệt lớp Ball chỉ có ở CWRU) mà không gây bất ổn loss.

## 3. Dữ liệu (không có trong repo này)

| Bộ dữ liệu | Tần số gốc | Kênh tín hiệu | Ghi chú |
|---|---|---|---|
| CWRU | 12 kHz | DE + FE | — |
| MFPT | 97,656 / 48,828 Hz (hỗn hợp) | trường `bearing.gs` | — |
| Paderborn | 64 kHz | `vibration_1` | 180 bản ghi (chỉ điều kiện N15_M07_F10) |

**Pipeline xử lý dữ liệu** (`process_data.py`):
```
Raw Data (.mat/.npz)
   → resample (nếu cần)
   → chia train/val/test theo TỪNG BẢN GHI TÍN HIỆU (không chia theo cửa sổ,
     để tránh rò rỉ dữ liệu giữa các tập — 2 cửa sổ từ cùng 1 bản ghi
     không bao giờ rơi vào 2 tập khác nhau)
   → lưu tín hiệu thô đã chia vào Data_test/
   → cắt cửa sổ trượt (sliding window) + chuẩn hóa z-score
   → lưu vào processed_data/ (tần số gốc từng bộ) và processed_data_std/ (đồng bộ về 12kHz cho cross-dataset)
```

Nhãn (label space):

| Label | CWRU | MFPT | Paderborn |
|---|---|---|---|
| 0 | Normal | Baseline | Normal (K001, K002, K006) |
| 1 | Inner Race | Inner Race | Inner Race (KI01, KI05, KI07) |
| 2 | Outer Race | Outer Race | Outer Race (KA01, KA05, KA07) |
| 3 | Ball | — | — |

Do các file dữ liệu (thô lẫn đã xử lý) tổng cộng **hơn 25GB** và có nhiều file `.npy` vượt 100MB, chúng không được đẩy lên GitHub. Để tái tạo, cần đặt dữ liệu gốc (CWRU/MFPT/Paderborn) vào thư mục `Raw Data/` rồi chạy:
```bash
python process_data.py --dataset all
```

## 4. Kết quả

### 4 mô hình riêng lẻ (train & test trên cùng bộ dữ liệu)

| Mô hình | Test Accuracy |
|---|---|
| CWRU | **98.9%** |
| MFPT | 74.2% |
| Paderborn | **99.6%** |
| Merged (12kHz, cả 3 bộ) | 97.2% |

### 9 kịch bản cross-dataset (S1–S9) — train trên 1 bộ, test trên bộ khác

| Kịch bản | Train → Test | Test Accuracy |
|---|---|---|
| S1 | CWRU → CWRU | 96.7% |
| S2 | CWRU → MFPT | 0.0% |
| S3 | CWRU → Paderborn | 34.4% |
| S4 | MFPT → CWRU | 35.9% |
| S5 | MFPT → MFPT | 100% |
| S6 | MFPT → Paderborn | 24.2% |
| S7 | Paderborn → CWRU | 14.9% |
| S8 | Paderborn → MFPT | 10.0% |
| S9 | Paderborn → Paderborn | 100% |

**Nhận xét:** mô hình đạt độ chính xác rất cao khi train/test trên cùng một bộ dữ liệu (S1, S5, S9 và 4 mô hình riêng lẻ), nhưng **giảm mạnh khi tổng quát hóa sang bộ dữ liệu khác** (S2–S4, S6–S8) — cho thấy domain shift giữa các bộ dữ liệu (khác cảm biến, khác điều kiện vận hành, khác loại vòng bi) là một thách thức lớn, đúng như thực trạng chung của bài toán transfer learning trong chẩn đoán hư hỏng công nghiệp. Đây cũng là động lực để khám phá các kỹ thuật domain adaptation trong các hướng phát triển tiếp theo.

## 5. Cấu trúc mã nguồn trong repo này

| File | Vai trò |
|---|---|
| `process_data.py` | Pipeline xử lý dữ liệu đầy đủ (không chạy được do thiếu `Raw Data/`, nhưng mã nguồn đầy đủ để tham khảo logic) |
| `extract_data.py` | `FeatureExtractor` — CNN1D + CNN2D-STFT (2 nhánh trích đặc trưng) |
| `train_model.py` | Kiến trúc `CNNGRU` + hàm `train()` cho 4 mô hình riêng lẻ |
| `run.py` | Script chạy huấn luyện individual/merged |
| `dashboard.py` | Giao diện Tkinter để test nhanh 1 mô hình đã huấn luyện, xem confusion matrix, báo cáo phân loại |
| `inference.py` | `OpenSetClassifier` — suy luận open-set: nếu độ tin cậy dự đoán thấp hơn ngưỡng τ, trả về nhãn "None" (chưa từng gặp) thay vì đoán bừa |
| `tools.py` | CLI tiện ích: kiểm tra tiến độ huấn luyện, xem nội dung file `.npy`/`.pth` |
| `saved_models/` | **Có trong repo** — checkpoint đã huấn luyện (`.pth`), kết quả (`training_results.json`), biểu đồ (confusion matrix, training curves) cho cả 13 kịch bản |

### Các thư mục **không có** trong repo (do dung lượng)

| Thư mục | Nội dung | Dung lượng |
|---|---|---|
| `Data/` | Dữ liệu gốc (raw) từ 3 bộ CWRU/MFPT/Paderborn | ~22 GB |
| `Data_test/` | Tín hiệu thô đã chia train/val/test theo bản ghi | ~880 MB |
| `processed_data/` | Cửa sổ đã cắt + z-score, tần số gốc từng bộ | ~1.8 GB |
| `processed_data_std/` | Giống trên nhưng đồng bộ 12kHz cho cross-dataset | ~560 MB |

## 6. Cách chạy (nếu có đủ dữ liệu gốc)

```bash
# 1. Xử lý dữ liệu
python process_data.py --dataset all

# 2. Train 4 mô hình riêng lẻ
python -u run.py --jobs individual --epochs 200
python -u run.py --jobs merged --epochs 200

# 3. Đánh giá bằng GUI
python dashboard.py

# 4. Đánh giá open-set qua CLI
python inference.py --dataset merged --tau 0.7
```

## 7. Tác giả

Đinh Đức Hiệp
