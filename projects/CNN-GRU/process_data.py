"""
Script bóc tách và xử lý dữ liệu bearing fault diagnosis
- CWRU Bearing Dataset (.npz) -> 4 nhãn: Normal(0), IR(1), OR(2), B(3) — DE12 + FE (@12kHz, bỏ DE48)
- MFPT Fault Data Sets (.mat) -> 3 nhãn: baseline(0), inner_race(1), outer_race(2)
- Paderborn (.mat) -> 3 nhãn: Normal(0), IR(1), OR(2)

Pipeline mỗi dataset:
    Đọc file → Resample về 12kHz
    → Split tín hiệu gốc (70/15/15, stratified) → Lưu tín hiệu gốc vào Data_test/
    → Sliding window từng split riêng → Z-score → Lưu vào processed_data/

Merged: gộp raw signals từ Data_test/{cwru,mfpt,paderborn}/ theo split,
        lưu Data_test/merged/, rồi window → processed_data/merged/.
"""

import os
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split
from collections import Counter


def print_dataset_table(title, X, y, label_names=None, n_show=10, threshold=20):
    n = len(y)
    n_show = min(n_show, n)
    print(f"{title}:")
    print("┌─────┬──────────────────────────┬───────┐")
    print("│ STT │ 2048 cột tín hiệu rung   │ Nhãn  │")
    print("├─────┼──────────────────────────┼───────┤")
    for i in range(n_show):
        arr = np.array2string(X[i], precision=3, separator=", ",
                              threshold=threshold, max_line_width=10**9)
        lbl = int(y[i])
        label_hint = f" <- {label_names[lbl]}" if label_names and lbl in label_names else ""
        print(f"│ {i+1:>3} │ {arr:<24} │ {lbl:>3} │{label_hint}")
    if n > n_show:
        print("│ ... │ ...                      │  ...  │")
        print(f"│{n:>4} │ [..{X.shape[1]} values..]         │  .. │")
    print("└─────┴──────────────────────────┴───────┘")


# CẤU HÌNH CHUNG
SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))

CWRU_DATA_DIR      = os.path.join(SOFTWARE_DIR, "Data", "CWRU_Bearing_NumPy-main", "Data")
MFPT_DATA_DIR      = os.path.join(SOFTWARE_DIR, "Data", "MFPT Fault Data Sets")
PADERBORN_DATA_DIR = os.path.join(SOFTWARE_DIR, "Data", "Paderborn")

OUTPUT_CWRU      = os.path.join(SOFTWARE_DIR, "processed_data", "cwru")
OUTPUT_MFPT      = os.path.join(SOFTWARE_DIR, "processed_data", "mfpt")
OUTPUT_PADERBORN = os.path.join(SOFTWARE_DIR, "processed_data", "paderborn")
OUTPUT_MERGED    = os.path.join(SOFTWARE_DIR, "processed_data", "merged")

DATA_TEST_CWRU      = os.path.join(SOFTWARE_DIR, "Data_test", "cwru")
DATA_TEST_MFPT      = os.path.join(SOFTWARE_DIR, "Data_test", "mfpt")
DATA_TEST_PADERBORN = os.path.join(SOFTWARE_DIR, "Data_test", "paderborn")
DATA_TEST_MERGED    = os.path.join(SOFTWARE_DIR, "Data_test", "merged")

WINDOW_SIZE  = 2048
STEP_SIZE    = 1024       # Overlap 50%
TARGET_FS    = 12_000     # Hz — tần số chuẩn cho merged + cross
MFPT_FS      = 48_828     # Hz — chuẩn nội bộ MFPT (downsample baseline/OR-2 từ 97,656Hz)
PADERBORN_FS = 64_000     # Hz — rate gốc Paderborn

OUTPUT_STD_CWRU      = os.path.join(SOFTWARE_DIR, "processed_data_std", "cwru")
OUTPUT_STD_MFPT      = os.path.join(SOFTWARE_DIR, "processed_data_std", "mfpt")
OUTPUT_STD_PADERBORN = os.path.join(SOFTWARE_DIR, "processed_data_std", "paderborn")

NUM_KNOWN_CLASSES = 4

LABEL_NAMES_CWRU      = {0: "Normal", 1: "IR",         2: "OR",         3: "B"}
LABEL_NAMES_MFPT      = {0: "Baseline", 1: "Inner Race", 2: "Outer Race"}
LABEL_NAMES_PADERBORN = {0: "Normal",   1: "Inner Race", 2: "Outer Race"}
LABEL_NAMES_MERGED    = {0: "Normal",   1: "IR",         2: "OR",         3: "Ball"}

PADERBORN_LABEL_MAP = {
    "K001": 0, "K002": 0, "K006": 0,
    "KI01": 1, "KI05": 1, "KI07": 1,
    "KA01": 2, "KA05": 2, "KA07": 2,
}
PADERBORN_OPERATING_CONDITION = "N15_M07_F10"


# HÀM TIỆN ÍCH — XỬ LÝ TÍN HIỆU
def get_label_from_cwru_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    fault_type = parts[1]
    if fault_type == "Normal":        return 0
    elif fault_type == "IR":          return 1
    elif fault_type.startswith("OR"): return 2
    elif fault_type == "B":           return 3
    return None


def resample_signal(signal: np.ndarray, src_fs: int, tgt_fs: int = TARGET_FS) -> np.ndarray:
    # Dùng resample_poly (polyphase) thay vì resample FFT: nhanh hơn và không giả định tín hiệu tuần hoàn
    # up = tgt_fs/gcd, down = src_fs/gcd — rút gọn tỉ lệ để tránh bộ lọc quá lớn
    if src_fs == tgt_fs:
        return signal.astype(np.float64)
    g = math.gcd(src_fs, tgt_fs)
    return resample_poly(signal.astype(np.float64), tgt_fs // g, src_fs // g)


def sliding_window(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    n_windows = (len(signal) - window_size) // step_size + 1
    windows = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        start = i * step_size
        windows[i] = signal[start:start + window_size]
    return windows


def zscore_normalize(X):
    means = X.mean(axis=1, keepdims=True)
    stds  = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds


def compute_class_weights_sqrt(y, num_classes=None):
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    counter = Counter(y.tolist())
    N, K = len(y), num_classes
    weights = np.zeros(K, dtype=np.float32)
    for k in range(K):
        n_k = counter.get(k, 1)
        weights[k] = float(np.sqrt(N / (K * n_k)))
    return weights


def print_distribution(labels, label_names, title, skipped=None, fp=None):
    counts = Counter(labels.tolist() if hasattr(labels, "tolist") else labels)
    total = sum(counts.values())
    lines = [f"\n  [{title}] Phân phối ({total} mẫu):"]
    for lbl in sorted(counts):
        n = counts[lbl]
        lines.append(f"    {label_names.get(lbl, str(lbl)):15s}: {n:6d}  ({n / total * 100:6.2f}%)")
    if skipped is not None:
        lines.append(f"  [{title}] Bị loại: {len(skipped)} file")
        for f in skipped:
            lines.append(f"    - {f}")
    text = "\n".join(lines)
    print(text)
    if fp is not None:
        fp.write(text + "\n")


def plot_samples(X, y, label_names, title, save_path):
    unique_labels = sorted(np.unique(y))
    n_labels = len(unique_labels)
    fig, axes = plt.subplots(n_labels, 1, figsize=(14, 3 * n_labels))
    if n_labels == 1:
        axes = [axes]
    for ax, label in zip(axes, unique_labels):
        idx = np.where(y == label)[0][0]
        ax.plot(X[idx], linewidth=0.5)
        ax.set_title(f"{label_names[label]} (label={label})", fontsize=12)
        ax.set_xlabel("Sample points")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# HÀM TIỆN ÍCH — SPLIT & LƯU
def split_signals(signals, labels, filenames=None, train=0.70, val=0.15, test=0.15):
    """Chia tập ở mức tín hiệu gốc (trước sliding window), per-class stratified.

    Tránh data leakage: các window từ cùng 1 tín hiệu không bị rơi vào
    cả train lẫn test như khi split sau sliding window.
    Đảm bảo mỗi lớp có ít nhất 1 recording trong mỗi split (kể cả lớp ít mẫu như MFPT Baseline).

    filenames: list tên file gốc tương ứng với mỗi signal (tuỳ chọn).
               Nếu truyền vào, trả về thêm fnames_tr, fnames_va, fnames_te.
    """
    rng = np.random.RandomState(42)
    lbl_arr = np.array(labels)
    idx_tr, idx_va, idx_te = [], [], []

    for cls in np.unique(lbl_arr):
        cls_idx = rng.permutation(np.where(lbl_arr == cls)[0])
        n = len(cls_idx)
        # Tính số lượng mỗi split, đảm bảo ít nhất 1 mẫu trong mỗi split
        n_te = max(1, round(n * test))
        n_va = max(1, round(n * val))
        n_tr = n - n_te - n_va
        if n_tr < 1:
            # Không đủ 3 recordings: ưu tiên train > val > test
            n_tr, n_va, n_te = max(1, n - 2), min(1, n - 1), 1
        idx_te.extend(cls_idx[:n_te])
        idx_va.extend(cls_idx[n_te:n_te + n_va])
        idx_tr.extend(cls_idx[n_te + n_va:])

    idx_tr = np.array(idx_tr)
    idx_va = np.array(idx_va)
    idx_te = np.array(idx_te)
    sigs_tr = [signals[i] for i in idx_tr]
    sigs_va = [signals[i] for i in idx_va]
    sigs_te = [signals[i] for i in idx_te]
    if filenames is not None:
        fnames_tr = [filenames[i] for i in idx_tr]
        fnames_va = [filenames[i] for i in idx_va]
        fnames_te = [filenames[i] for i in idx_te]
    else:
        fnames_tr = fnames_va = fnames_te = None
    return (sigs_tr, lbl_arr[idx_tr], sigs_va, lbl_arr[idx_va], sigs_te, lbl_arr[idx_te],
            fnames_tr, fnames_va, fnames_te)


def save_raw_signals(out_dir, sigs_tr, y_tr, sigs_va, y_va, sigs_te, y_te,
                     fnames_tr=None, fnames_va=None, fnames_te=None):
    """Lưu tín hiệu gốc (variable length) vào Data_test/ dưới dạng numpy object array.

    Nếu truyền fnames_*, lưu thêm filenames_{split}.npy để truy xuất file gốc.
    """
    os.makedirs(out_dir, exist_ok=True)
    for split_name, sigs, y, fnames in [
        ("train", sigs_tr, y_tr, fnames_tr),
        ("val",   sigs_va, y_va, fnames_va),
        ("test",  sigs_te, y_te, fnames_te),
    ]:
        arr = np.empty(len(sigs), dtype=object)
        for i, s in enumerate(sigs):
            arr[i] = np.array(s, dtype=np.float64)
        np.save(os.path.join(out_dir, f"signals_{split_name}.npy"), arr, allow_pickle=True)
        np.save(os.path.join(out_dir, f"labels_{split_name}.npy"),  np.array(y))
        if fnames is not None:
            np.save(os.path.join(out_dir, f"filenames_{split_name}.npy"), np.array(fnames, dtype=str))
    print(f"  [Data_test] Đã lưu → {out_dir}")
    print(f"    train: {len(sigs_tr)} signals | val: {len(sigs_va)} | test: {len(sigs_te)}")


def load_raw_signals(raw_dir, split):
    """Tải tín hiệu gốc từ Data_test/. Trả về (sigs, y, filenames).
    filenames là None nếu chưa có filenames_{split}.npy (data cũ chưa chạy lại).
    """
    sigs  = np.load(os.path.join(raw_dir, f"signals_{split}.npy"), allow_pickle=True)
    y     = np.load(os.path.join(raw_dir, f"labels_{split}.npy"))
    fpath = os.path.join(raw_dir, f"filenames_{split}.npy")
    fnames = list(np.load(fpath)) if os.path.exists(fpath) else None
    return list(sigs), y, fnames


def signals_to_windows(signals, labels):
    """Áp dụng sliding window lên danh sách tín hiệu, trả về (X, y) dạng 2D array."""
    X_wins, y_wins = [], []
    for sig, lbl in zip(signals, labels):
        wins = sliding_window(sig)
        X_wins.append(wins)
        y_wins.extend([int(lbl)] * len(wins))
    return np.vstack(X_wins), np.array(y_wins)


def save_dataset(out_dir, X_train, y_train, X_val, y_val, X_test, y_test,
                 num_classes, label_names=None):
    """Z-score per-sample + lưu windows đã split vào processed_data/."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"),       zscore_normalize(X_train))
    np.save(os.path.join(out_dir, "Y_train.npy"),       y_train)
    np.save(os.path.join(out_dir, "X_val.npy"),         zscore_normalize(X_val))
    np.save(os.path.join(out_dir, "Y_val.npy"),         y_val)
    np.save(os.path.join(out_dir, "X_test.npy"),        zscore_normalize(X_test))
    np.save(os.path.join(out_dir, "Y_test.npy"),        y_test)
    weights = compute_class_weights_sqrt(y_train, num_classes=num_classes)
    np.save(os.path.join(out_dir, "class_weights.npy"), weights)
    print(f"  [{os.path.basename(out_dir)}] Class counts (train): {dict(Counter(y_train.tolist()))}")
    print(f"  [{os.path.basename(out_dir)}] Class weights (sqrt): {weights.tolist()}")


def extract_paderborn_vibration(mat_data):
    """Trích xuất tín hiệu vibration_1 từ file .mat Paderborn."""
    keys = [k for k in mat_data.keys() if not k.startswith("__")]
    if len(keys) == 0:
        raise KeyError("Không có key dữ liệu hợp lệ trong file .mat")
    root = mat_data[keys[0]][0, 0]
    y_struct = root["Y"]
    for i in range(y_struct.shape[1]):
        name_str = str(np.array(y_struct["Name"][0, i]).squeeze()).lower()
        if "vibration_1" in name_str:
            return np.array(y_struct["Data"][0, i]).flatten()
    raise KeyError("Không tìm thấy kênh 'vibration_1'")


# PHẦN 1: CWRU — DE12 + FE @12kHz (bỏ DE48)
def process_cwru():
    all_signals, all_labels, all_filenames, skipped_files = [], [], [], []

    for rpm_folder in sorted(os.listdir(CWRU_DATA_DIR)):
        rpm_path = os.path.join(CWRU_DATA_DIR, rpm_folder)
        if not os.path.isdir(rpm_path):
            continue
        npz_files = sorted([f for f in os.listdir(rpm_path)
                             if f.endswith(".npz") and "DE48" not in f])
        for npz_file in npz_files:
            label = get_label_from_cwru_filename(npz_file)
            if label is None:
                skipped_files.append(npz_file)
                continue
            try:
                data = np.load(os.path.join(rpm_path, npz_file))
            except Exception:
                skipped_files.append(npz_file)
                continue
            for ch in ("DE", "FE"):
                if ch not in data:
                    continue
                signal = data[ch].flatten()
                if len(signal) < WINDOW_SIZE:
                    continue
                if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                    continue
                all_signals.append(signal.astype(np.float64))
                all_labels.append(label)
                all_filenames.append(f"{rpm_folder}/{npz_file}::{ch}")

    # Lưu phân phối raw
    os.makedirs(OUTPUT_CWRU, exist_ok=True)
    with open(os.path.join(OUTPUT_CWRU, "distribution.txt"), "w", encoding="utf-8") as fp:
        print_distribution(all_labels, LABEL_NAMES_CWRU,
                           "CWRU raw signals @12kHz (DE12+FE)", skipped=skipped_files, fp=fp)

    # Plot raw samples (1 mẫu/nhãn, 5000 pts)
    plot_dir = os.path.join(OUTPUT_CWRU, "plots")
    sample_per_label = {}
    for sig, lbl in zip(all_signals, all_labels):
        sample_per_label.setdefault(lbl, sig)
    raw_X = np.array([sample_per_label[l][:5000] for l in sorted(sample_per_label)])
    raw_y = np.array(sorted(sample_per_label))
    plot_samples(raw_X, raw_y, LABEL_NAMES_CWRU,
                 "CWRU - Raw Signal Samples (DE12 + FE, @12kHz)",
                 os.path.join(plot_dir, "cwru_raw_samples.png"))

    # Split tại mức tín hiệu gốc
    sigs_tr, y_tr_raw, sigs_va, y_va_raw, sigs_te, y_te_raw, fnames_tr, fnames_va, fnames_te = \
        split_signals(all_signals, all_labels, filenames=all_filenames)

    # Lưu tín hiệu gốc vào Data_test/cwru/
    save_raw_signals(DATA_TEST_CWRU,
                     sigs_tr, y_tr_raw, sigs_va, y_va_raw, sigs_te, y_te_raw,
                     fnames_tr, fnames_va, fnames_te)

    # Sliding window từng split
    X_train, y_train = signals_to_windows(sigs_tr, y_tr_raw)
    X_val,   y_val   = signals_to_windows(sigs_va, y_va_raw)
    X_test,  y_test  = signals_to_windows(sigs_te, y_te_raw)

    with open(os.path.join(OUTPUT_CWRU, "distribution.txt"), "a", encoding="utf-8") as fp:
        print_distribution(y_train, LABEL_NAMES_CWRU, "CWRU train (sau sliding window)", fp=fp)
        print_distribution(y_val,   LABEL_NAMES_CWRU, "CWRU val   (sau sliding window)", fp=fp)
        print_distribution(y_test,  LABEL_NAMES_CWRU, "CWRU test  (sau sliding window)", fp=fp)

    # Z-score + lưu vào processed_data/cwru/
    save_dataset(OUTPUT_CWRU, X_train, y_train, X_val, y_val, X_test, y_test,
                 num_classes=4, label_names=LABEL_NAMES_CWRU)

    X_train_plot = zscore_normalize(X_train)
    plot_samples(X_train_plot, y_train, LABEL_NAMES_CWRU,
                 "CWRU - Processed Samples (Z-score, window=2048, @12kHz)",
                 os.path.join(plot_dir, "cwru_processed_samples.png"))

    return (zscore_normalize(X_train), y_train,
            zscore_normalize(X_val),   y_val,
            zscore_normalize(X_test),  y_test)


# PHẦN 2: MFPT — chuẩn hoá nội bộ về MFPT_FS (48,828Hz)
def process_mfpt():
    folder_label_map = {
        "1 - Three Baseline Conditions":              0,
        "4 - Seven Inner Race Fault Conditions":      1,
        "2 - Three Outer Race Fault Conditions":      2,
        "3 - Seven More Outer Race Fault Conditions": 2,
    }
    all_signals, all_labels, all_filenames, skipped_files = [], [], [], []

    for folder, label in folder_label_map.items():
        folder_path = os.path.join(MFPT_DATA_DIR, folder)
        if not os.path.exists(folder_path):
            continue
        for mat_file in sorted(f for f in os.listdir(folder_path) if f.endswith(".mat")):
            try:
                data = sio.loadmat(os.path.join(folder_path, mat_file))
                if "bearing" not in data:
                    skipped_files.append(mat_file)
                    continue
                bearing = data["bearing"]
                gs     = bearing[0, 0]["gs"].flatten()
                src_fs = int(bearing[0, 0]["sr"].flatten()[0])
                if len(gs) < WINDOW_SIZE:
                    skipped_files.append(mat_file)
                    continue
                if np.any(np.isnan(gs)) or np.any(np.isinf(gs)):
                    skipped_files.append(mat_file)
                    continue
                orig_len = len(gs)
                # Chuẩn hóa nội bộ MFPT: baseline và OR-2 ghi ở 97,656Hz → downsample về 48,828Hz
                # để toàn bộ MFPT dùng cùng 1 sample rate trước khi cắt window
                gs = resample_signal(gs, src_fs, MFPT_FS)
                if src_fs != MFPT_FS:
                    print(f"  {mat_file}: {src_fs} Hz → {MFPT_FS} Hz  "
                          f"({orig_len} → {len(gs)} samples)")
                all_signals.append(gs)
                all_labels.append(label)
                all_filenames.append(f"{folder}/{mat_file}")
            except Exception:
                skipped_files.append(mat_file)

    os.makedirs(OUTPUT_MFPT, exist_ok=True)
    with open(os.path.join(OUTPUT_MFPT, "distribution.txt"), "w", encoding="utf-8") as fp:
        print_distribution(all_labels, LABEL_NAMES_MFPT,
                           f"MFPT raw signals @{MFPT_FS}Hz (within-MFPT std)", skipped=skipped_files, fp=fp)

    plot_dir = os.path.join(OUTPUT_MFPT, "plots")
    sample_per_label = {}
    for sig, lbl in zip(all_signals, all_labels):
        sample_per_label.setdefault(lbl, sig)
    raw_X = np.array([sample_per_label[l][:5000] for l in sorted(sample_per_label)])
    raw_y = np.array(sorted(sample_per_label))
    plot_samples(raw_X, raw_y, LABEL_NAMES_MFPT,
                 f"MFPT - Raw Signal Samples (@{MFPT_FS}Hz)",
                 os.path.join(plot_dir, "mfpt_raw_samples.png"))

    # Split tại mức tín hiệu gốc
    sigs_tr, y_tr_raw, sigs_va, y_va_raw, sigs_te, y_te_raw, fnames_tr, fnames_va, fnames_te = \
        split_signals(all_signals, all_labels, filenames=all_filenames)

    save_raw_signals(DATA_TEST_MFPT,
                     sigs_tr, y_tr_raw, sigs_va, y_va_raw, sigs_te, y_te_raw,
                     fnames_tr, fnames_va, fnames_te)

    X_train, y_train = signals_to_windows(sigs_tr, y_tr_raw)
    X_val,   y_val   = signals_to_windows(sigs_va, y_va_raw)
    X_test,  y_test  = signals_to_windows(sigs_te, y_te_raw)

    with open(os.path.join(OUTPUT_MFPT, "distribution.txt"), "a", encoding="utf-8") as fp:
        print_distribution(y_train, LABEL_NAMES_MFPT, "MFPT train (sau sliding window)", fp=fp)
        print_distribution(y_val,   LABEL_NAMES_MFPT, "MFPT val   (sau sliding window)", fp=fp)
        print_distribution(y_test,  LABEL_NAMES_MFPT, "MFPT test  (sau sliding window)", fp=fp)

    save_dataset(OUTPUT_MFPT, X_train, y_train, X_val, y_val, X_test, y_test,
                 num_classes=3, label_names=LABEL_NAMES_MFPT)

    X_train_plot = zscore_normalize(X_train)
    plot_samples(X_train_plot, y_train, LABEL_NAMES_MFPT,
                 f"MFPT - Processed Samples (Z-score, window=2048, @{MFPT_FS}Hz)",
                 os.path.join(plot_dir, "mfpt_processed_samples.png"))

    return (zscore_normalize(X_train), y_train,
            zscore_normalize(X_val),   y_val,
            zscore_normalize(X_test),  y_test)


# PHẦN 3: PADERBORN — vibration_1 @64kHz (giữ nguyên, không resample)
def process_paderborn():
    all_signals, all_labels, all_filenames, skipped_files = [], [], [], []

    for bearing_code, label in PADERBORN_LABEL_MAP.items():
        bearing_path = os.path.join(PADERBORN_DATA_DIR, bearing_code)
        if not os.path.isdir(bearing_path):
            print(f"[Paderborn] Không tìm thấy thư mục: {bearing_code}")
            continue
        mat_files = sorted([
            f for f in os.listdir(bearing_path)
            if f.endswith(".mat") and PADERBORN_OPERATING_CONDITION in f
        ])
        for mat_file in mat_files:
            try:
                data   = sio.loadmat(os.path.join(bearing_path, mat_file))
                signal = extract_paderborn_vibration(data)
                if len(signal) >= 256_000:
                    signal = signal[:256_000]
                if len(signal) < WINDOW_SIZE:
                    skipped_files.append(mat_file)
                    continue
                if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                    skipped_files.append(mat_file)
                    continue
                all_signals.append(signal)
                all_labels.append(label)
                all_filenames.append(f"{bearing_code}/{mat_file}")
            except Exception:
                skipped_files.append(mat_file)

    if not all_signals:
        raise RuntimeError("Không đọc được dữ liệu Paderborn hợp lệ.")

    os.makedirs(OUTPUT_PADERBORN, exist_ok=True)
    with open(os.path.join(OUTPUT_PADERBORN, "distribution.txt"), "w", encoding="utf-8") as fp:
        print_distribution(all_labels, LABEL_NAMES_PADERBORN,
                           f"Paderborn raw signals @{PADERBORN_FS}Hz (native)", skipped=skipped_files, fp=fp)

    plot_dir = os.path.join(OUTPUT_PADERBORN, "plots")
    sample_per_label = {}
    for sig, lbl in zip(all_signals, all_labels):
        sample_per_label.setdefault(lbl, sig)
    raw_X = np.array([sample_per_label[l][:5000] for l in sorted(sample_per_label)])
    raw_y = np.array(sorted(sample_per_label))
    plot_samples(raw_X, raw_y, LABEL_NAMES_PADERBORN,
                 f"Paderborn - Raw Signal Samples (vibration_1, @{PADERBORN_FS}Hz)",
                 os.path.join(plot_dir, "paderborn_raw_samples.png"))

    # Split tại mức tín hiệu gốc
    sigs_tr, y_tr_raw, sigs_va, y_va_raw, sigs_te, y_te_raw, fnames_tr, fnames_va, fnames_te = \
        split_signals(all_signals, all_labels, filenames=all_filenames)

    save_raw_signals(DATA_TEST_PADERBORN,
                     sigs_tr, y_tr_raw, sigs_va, y_va_raw, sigs_te, y_te_raw,
                     fnames_tr, fnames_va, fnames_te)

    X_train, y_train = signals_to_windows(sigs_tr, y_tr_raw)
    X_val,   y_val   = signals_to_windows(sigs_va, y_va_raw)
    X_test,  y_test  = signals_to_windows(sigs_te, y_te_raw)

    with open(os.path.join(OUTPUT_PADERBORN, "distribution.txt"), "a", encoding="utf-8") as fp:
        print_distribution(y_train, LABEL_NAMES_PADERBORN, "Paderborn train (sau sliding window)", fp=fp)
        print_distribution(y_val,   LABEL_NAMES_PADERBORN, "Paderborn val   (sau sliding window)", fp=fp)
        print_distribution(y_test,  LABEL_NAMES_PADERBORN, "Paderborn test  (sau sliding window)", fp=fp)

    save_dataset(OUTPUT_PADERBORN, X_train, y_train, X_val, y_val, X_test, y_test,
                 num_classes=3, label_names=LABEL_NAMES_PADERBORN)

    X_train_plot = zscore_normalize(X_train)
    plot_samples(X_train_plot, y_train, LABEL_NAMES_PADERBORN,
                 f"Paderborn - Processed Samples (Z-score, window=2048, @{PADERBORN_FS}Hz)",
                 os.path.join(plot_dir, "paderborn_processed_samples.png"))

    return (zscore_normalize(X_train), y_train,
            zscore_normalize(X_val),   y_val,
            zscore_normalize(X_test),  y_test)


# PHẦN 4: MERGED — gộp raw từ Data_test/, window → processed_data/merged/
def merge_datasets():
    """Gộp tín hiệu gốc từ Data_test/{cwru,mfpt,paderborn}/ theo split (train/val/test),
    lưu vào Data_test/merged/, rồi sliding window + z-score → processed_data/merged/.

    Giữ nguyên ranh giới train/val/test của từng dataset con: không re-split lại,
    tránh tín hiệu từ split test của 1 dataset lọt vào train của merged.
    """
    # (raw_dir, src_fs) — mỗi dataset lưu ở rate khác nhau, resample → 12kHz khi merge
    raw_dirs_fs = [
        (DATA_TEST_CWRU,      TARGET_FS),    # 12kHz — không đổi
        (DATA_TEST_MFPT,      MFPT_FS),      # 48,828Hz → 12kHz
        (DATA_TEST_PADERBORN, PADERBORN_FS), # 64,000Hz → 12kHz
    ]

    sigs_tr_all, y_tr_all, fnames_tr_all = [], [], []
    sigs_va_all, y_va_all, fnames_va_all = [], [], []
    sigs_te_all, y_te_all, fnames_te_all = [], [], []

    for raw_dir, src_fs in raw_dirs_fs:
        if not os.path.isdir(raw_dir):
            print(f"[Merge] Bỏ qua {raw_dir} (chưa có folder)")
            continue
        ds_name = os.path.basename(raw_dir)
        try:
            sigs_tr, y_tr, fnames_tr = load_raw_signals(raw_dir, "train")
            sigs_va, y_va, fnames_va = load_raw_signals(raw_dir, "val")
            sigs_te, y_te, fnames_te = load_raw_signals(raw_dir, "test")
            if src_fs != TARGET_FS:
                # Đồng bộ sample rate: MFPT (48,828Hz) và Paderborn (64kHz) → 12kHz
                # để tín hiệu từ 3 dataset có cùng độ phân giải thời gian khi gộp vào merged
                print(f"  [Merge] Resample {ds_name}: {src_fs}Hz → {TARGET_FS}Hz")
                sigs_tr = [resample_signal(s, src_fs, TARGET_FS) for s in sigs_tr]
                sigs_va = [resample_signal(s, src_fs, TARGET_FS) for s in sigs_va]
                sigs_te = [resample_signal(s, src_fs, TARGET_FS) for s in sigs_te]
            sigs_tr_all.extend(sigs_tr); y_tr_all.extend(y_tr.tolist())
            sigs_va_all.extend(sigs_va); y_va_all.extend(y_va.tolist())
            sigs_te_all.extend(sigs_te); y_te_all.extend(y_te.tolist())
            # Prefix tên file với tên dataset để phân biệt nguồn trong merged
            if fnames_tr is not None:
                fnames_tr_all.extend(f"{ds_name}::{fn}" for fn in fnames_tr)
                fnames_va_all.extend(f"{ds_name}::{fn}" for fn in fnames_va)
                fnames_te_all.extend(f"{ds_name}::{fn}" for fn in fnames_te)
            print(f"  [Merge] Load {ds_name}: "
                  f"train={len(sigs_tr)} | val={len(sigs_va)} | test={len(sigs_te)}")
        except FileNotFoundError as e:
            print(f"[Merge] Bỏ qua {raw_dir} (thiếu file): {e}")

    if not sigs_tr_all:
        raise RuntimeError("Không có dataset nào để gộp. Chạy process_cwru/mfpt/paderborn trước.")

    # Lưu raw merged vào Data_test/merged/
    save_raw_signals(DATA_TEST_MERGED,
                     sigs_tr_all, np.array(y_tr_all),
                     sigs_va_all, np.array(y_va_all),
                     sigs_te_all, np.array(y_te_all),
                     fnames_tr_all or None, fnames_va_all or None, fnames_te_all or None)

    # Sliding window từng split
    X_train, y_train = signals_to_windows(sigs_tr_all, y_tr_all)
    X_val,   y_val   = signals_to_windows(sigs_va_all, y_va_all)
    X_test,  y_test  = signals_to_windows(sigs_te_all, y_te_all)

    # Shuffle train (seed cố định)
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    weights = compute_class_weights_sqrt(y_train, num_classes=NUM_KNOWN_CLASSES)

    os.makedirs(OUTPUT_MERGED, exist_ok=True)
    np.save(os.path.join(OUTPUT_MERGED, "X_train.npy"),       zscore_normalize(X_train))
    np.save(os.path.join(OUTPUT_MERGED, "Y_train.npy"),       y_train)
    np.save(os.path.join(OUTPUT_MERGED, "X_val.npy"),         zscore_normalize(X_val))
    np.save(os.path.join(OUTPUT_MERGED, "Y_val.npy"),         y_val)
    np.save(os.path.join(OUTPUT_MERGED, "X_test.npy"),        zscore_normalize(X_test))
    np.save(os.path.join(OUTPUT_MERGED, "Y_test.npy"),        y_test)
    np.save(os.path.join(OUTPUT_MERGED, "class_weights.npy"), weights)

    Y_all = np.concatenate([y_train, y_val, y_test])
    print(f"\n[Merged] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    with open(os.path.join(OUTPUT_MERGED, "distribution.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"Train shape: {X_train.shape}\n")
        fp.write(f"Val shape:   {X_val.shape}\n")
        fp.write(f"Test shape:  {X_test.shape}\n")
        print_distribution(Y_all,   LABEL_NAMES_MERGED, "Merged toàn bộ (train+val+test)", fp=fp)
        print_distribution(y_train, LABEL_NAMES_MERGED, "Merged tập train",                fp=fp)
        fp.write(f"\n  Class weights (sqrt): {weights.tolist()}\n")
    print(f"  Class weights (sqrt): {weights.tolist()}")

    return (zscore_normalize(X_train), y_train,
            zscore_normalize(X_val),   y_val,
            zscore_normalize(X_test),  y_test)


# PHẦN 5: CROSS PREP — resample tất cả về 12kHz cho train_cross.py
def _build_std_split(raw_dir, src_fs, out_dir, num_classes, label_names, ds_name):
    """Load raw signals từ Data_test/, resample → 12kHz, window, z-score, lưu vào processed_data_std/."""
    splits = {}
    for split in ("train", "val", "test"):
        sigs, y, _ = load_raw_signals(raw_dir, split)
        if src_fs != TARGET_FS:
            # Resample về 12kHz để tất cả 3 dataset có cùng rate cho cross-dataset training
            sigs = [resample_signal(s, src_fs, TARGET_FS) for s in sigs]
        X, y_win = signals_to_windows(sigs, y)
        splits[split] = (X, y_win)

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    save_dataset(out_dir, X_train, y_train, X_val, y_val, X_test, y_test,
                 num_classes=num_classes, label_names=label_names)
    print(f"  [{ds_name} @{TARGET_FS}Hz] → {out_dir}")


def process_for_cross():
    """Chuẩn bị processed_data_std/ cho 9 cross scenarios (tất cả @12kHz).

    CWRU: đã 12kHz → copy window trực tiếp.
    MFPT: load từ Data_test/mfpt/ @48,828Hz → resample → 12kHz.
    Paderborn: load từ Data_test/paderborn/ @64kHz → resample → 12kHz.
    """
    print("\n=== Chuẩn bị processed_data_std/ (12kHz) cho cross scenarios ===")

    # CWRU: src_fs = TARGET_FS → không resample
    _build_std_split(DATA_TEST_CWRU, TARGET_FS,
                     OUTPUT_STD_CWRU, 4, LABEL_NAMES_CWRU, "CWRU")

    # MFPT: 48,828Hz → 12kHz
    _build_std_split(DATA_TEST_MFPT, MFPT_FS,
                     OUTPUT_STD_MFPT, 3, LABEL_NAMES_MFPT, "MFPT")

    # Paderborn: 64,000Hz → 12kHz
    _build_std_split(DATA_TEST_PADERBORN, PADERBORN_FS,
                     OUTPUT_STD_PADERBORN, 3, LABEL_NAMES_PADERBORN, "Paderborn")

    print("  [cross_prep] Hoàn tất → processed_data_std/")


# MAIN
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Xử lý dữ liệu bearing fault diagnosis")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["cwru", "mfpt", "paderborn", "merged", "cross_prep", "all"],
        help=(
            "Dataset cần xử lý. 'all' = cwru+mfpt+paderborn+merged+cross_prep. "
            "'cross_prep' = tạo processed_data_std/ @12kHz cho train_cross.py."
        ),
    )
    args = parser.parse_args()

    if args.dataset in ["cwru", "all"]:
        cwru_results = process_cwru()
        print_dataset_table("CWRU Train", cwru_results[0], cwru_results[1], LABEL_NAMES_CWRU)
        print_dataset_table("CWRU Val",   cwru_results[2], cwru_results[3], LABEL_NAMES_CWRU)
        print_dataset_table("CWRU Test",  cwru_results[4], cwru_results[5], LABEL_NAMES_CWRU)

    if args.dataset in ["mfpt", "all"]:
        mfpt_results = process_mfpt()
        print_dataset_table("MFPT Train", mfpt_results[0], mfpt_results[1], LABEL_NAMES_MFPT)
        print_dataset_table("MFPT Val",   mfpt_results[2], mfpt_results[3], LABEL_NAMES_MFPT)
        print_dataset_table("MFPT Test",  mfpt_results[4], mfpt_results[5], LABEL_NAMES_MFPT)

    if args.dataset in ["paderborn", "all"]:
        paderborn_results = process_paderborn()
        print_dataset_table("Paderborn Train", paderborn_results[0], paderborn_results[1], LABEL_NAMES_PADERBORN)
        print_dataset_table("Paderborn Val",   paderborn_results[2], paderborn_results[3], LABEL_NAMES_PADERBORN)
        print_dataset_table("Paderborn Test",  paderborn_results[4], paderborn_results[5], LABEL_NAMES_PADERBORN)

    if args.dataset in ["merged", "all"]:
        merged_results = merge_datasets()
        print_dataset_table("Merged Train", merged_results[0], merged_results[1], LABEL_NAMES_MERGED)
        print_dataset_table("Merged Val",   merged_results[2], merged_results[3], LABEL_NAMES_MERGED)
        print_dataset_table("Merged Test",  merged_results[4], merged_results[5], LABEL_NAMES_MERGED)

    if args.dataset in ["cross_prep", "all"]:
        process_for_cross()
