

import io
import os
import threading
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("Agg")   # render không cần màn hình (dùng trong thread)
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageTk
from sklearn.metrics import classification_report, confusion_matrix

from train_model import CNNGRU
from process_data import (
    sliding_window, zscore_normalize, resample_signal,
    TARGET_FS, MFPT_FS, PADERBORN_FS,
)

# Sample rate mỗi model được train theo (process_data.py) — dùng để tự resample
# tín hiệu thô về đúng rate model mong đợi trước khi đưa vào sliding_window.
FS_BY_DATASET = {
    "cwru":      TARGET_FS,     # 12kHz native
    "mfpt":      MFPT_FS,       # 48,828Hz (chuẩn nội bộ MFPT)
    "paderborn": PADERBORN_FS,  # 64kHz native
    "merged":    TARGET_FS,     # 12kHz (đã resample lúc gộp)
}

SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tên nhãn hiển thị theo từng dataset
CLASS_NAMES = {
    "cwru":      ["Normal", "IR (Inner)", "OR (Outer)", "B (Ball)"],
    "mfpt":      ["Baseline", "Inner Race", "Outer Race"],
    "paderborn": ["Normal", "Inner Race", "Outer Race"],
    "merged":    ["Normal", "IR", "OR", "Ball"],
}

# Chỉ hiện 4 model individual trong dropdown, ẩn 9 cross scenarios
_INDIVIDUAL_MODELS = {"cwru", "mfpt", "paderborn", "merged"}


# QUÉT NGUỒN DATA & MODEL

def scan_models():
    """Quét saved_models/ → dict {tên hiển thị: đường dẫn .pth}.
    Chỉ lấy 4 model individual (cwru/mfpt/paderborn/merged), bỏ qua cross scenarios.
    """
    result = {}
    saved_dir = os.path.join(SOFTWARE_DIR, "saved_models")
    if not os.path.isdir(saved_dir):
        return result
    for run in sorted(os.listdir(saved_dir)):
        if run not in _INDIVIDUAL_MODELS:
            continue
        run_dir = os.path.join(saved_dir, run)
        if not os.path.isdir(run_dir):
            continue
        for f in sorted(os.listdir(run_dir)):
            # Bỏ checkpoint_latest.pth — chỉ lấy best model
            if f.endswith(".pth") and not f.startswith("checkpoint"):
                result[f"{run} / {f}"] = os.path.join(run_dir, f)
    return result


def scan_datasets():
    """Quét Data_test/ → dict {tên hiển thị: (path, ds_id)}.

    Chỉ hiện folder có signals_test.npy + filenames_test.npy (tín hiệu thô split
    test, được tạo khi chạy process_data.py --dataset all).
    """
    result = {}

    data_test_base = os.path.join(SOFTWARE_DIR, "Data_test")
    if os.path.isdir(data_test_base):
        for ds in sorted(os.listdir(data_test_base)):
            ds_dir = os.path.join(data_test_base, ds)
            if (os.path.exists(os.path.join(ds_dir, "signals_test.npy")) and
                    os.path.exists(os.path.join(ds_dir, "filenames_test.npy"))):
                key = f"Data_test / {ds}  [raw signal + filename]"
                result[key] = (ds_dir, ds)

    return result


# PIPELINE INFERENCE

def _predict_raw(model, signals, labels, num_classes, tau, device, filenames=None):
    """Inference pipeline trên tín hiệu thô — giống inference.py thực tế.

    Với mỗi tín hiệu:
      1. sliding_window(2048, stride=1024) → N windows
      2. zscore_normalize per window
      3. model predict → softmax → p_max per window
      4. window nào p_max < tau → vote None (-1)
      5. vote đa số → nhãn cuối của tín hiệu

    Trả về:
      all_preds : (N_sig,) nhãn dự đoán (-1 = None)
      all_true  : (N_sig,) nhãn thật
      details   : list dict mỗi tín hiệu, dùng để hiển thị bảng chi tiết
    """
    all_preds, all_true, details = [], [], []
    model.eval()
    with torch.no_grad():
        for i, (sig, label) in enumerate(zip(signals, labels)):
            sig = np.array(sig, dtype=np.float64)
            if len(sig) < 2048:
                continue  # tín hiệu quá ngắn, không đủ 1 window

            windows = sliding_window(sig)
            windows = zscore_normalize(windows)

            x      = torch.tensor(windows, dtype=torch.float32).to(device)
            logits = model(x)                        # (N_win, num_classes)
            probs  = torch.softmax(logits, dim=1)
            p_max, preds = probs.max(dim=1)          # confidence và class mỗi window

            # Open-set: window không đủ tự tin → không được tính vào vote
            preds[p_max < tau] = -1

            # Vote đa số để ra nhãn cuối của cả tín hiệu
            votes = {}
            for p in preds.cpu().numpy():
                votes[int(p)] = votes.get(int(p), 0) + 1
            final_pred = max(votes, key=votes.get)

            all_preds.append(final_pred)
            all_true.append(int(label))
            details.append({
                "filename": (filenames[i] if filenames is not None else f"signal_{i+1}"),
                "true":     int(label),
                "pred":     final_pred,
                "conf":     float(p_max.mean().item()),  # trung bình confidence tất cả windows
                "n_win":    len(windows),
            })

    return np.array(all_preds), np.array(all_true), details


def _build_cm_image(all_true, all_preds, cnames):
    """Vẽ confusion matrix dạng phần trăm, trả về PIL Image.

    Chỉ vẽ các nhãn thực sự xuất hiện trong tập test (bỏ qua None = -1).
    Nếu model predict nhãn ngoài cnames (ví dụ model merged 4 class test trên
    MFPT 3 class), nhãn đó hiển thị là "Class N" thay vì crash.
    """
    valid   = all_preds >= 0   # bỏ các dự đoán None
    present = sorted(set(all_true[valid].tolist()) | set(all_preds[valid].tolist()))
    pnames  = [(cnames[i] if i < len(cnames) else f"Class {i}") for i in present]
    cm      = confusion_matrix(all_true[valid], all_preds[valid], labels=present)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    n       = len(present)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(pnames, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(pnames, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion Matrix (%)", fontsize=11)
    for i in range(n):
        for j in range(n):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]})",
                    ha="center", va="center", fontsize=8, color=color)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _make_detail_table(details, cnames):
    """Tạo bảng text chi tiết từng tín hiệu.

    Cột: File (tên file gốc) | True (nhãn thật) | Pred (nhãn dự đoán)
         | Conf (confidence trung bình) | Win (số windows) | OK/ERR
    """
    lines = []
    # Tự điều chỉnh độ rộng cột File theo tên file dài nhất (tối đa 50 ký tự)
    w_fn   = max(min(50, max((len(d["filename"]) for d in details), default=10)), 10)
    header = f"  {'File':<{w_fn}}  {'True':<14}  {'Pred':<14}  {'Conf':>6}  {'Win':>4}  "
    sep    = "─" * len(header)
    lines.extend([sep, header, sep])

    for d in details:
        # Cắt tên file từ bên trái nếu quá dài (giữ phần cuối chứa tên file quan trọng)
        fn        = d["filename"][-w_fn:].ljust(w_fn)
        true_name = (cnames[d["true"]] if 0 <= d["true"] < len(cnames) else str(d["true"]))[:14]
        pred_name = ("None" if d["pred"] == -1
                     else (cnames[d["pred"]] if d["pred"] < len(cnames) else str(d["pred"])))[:14]
        ok        = "OK " if d["pred"] == d["true"] else "ERR"
        lines.append(f"  {fn}  {true_name:<14}  {pred_name:<14}  {d['conf']:>5.1%}  {d['n_win']:>4}  {ok}")

    lines.append(sep)
    return "\n".join(lines)


# DASHBOARD UI

class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bearing Fault Diagnosis — Dashboard")
        self.geometry("1150x750")
        self.minsize(900, 600)

        self.models   = scan_models()
        self.datasets = scan_datasets()
        self._cm_photo    = None   # giữ reference để Tkinter không GC mất ảnh
        self._cm_img_orig = None   # PIL Image gốc, dùng để resize lại khi cửa sổ thay đổi

        self._build_ui()

    def _build_ui(self):
        ttk.Label(self, text="Bearing Fault Diagnosis — Model Evaluation",
                  font=("Segoe UI", 14, "bold")).pack(pady=(12, 4))

        # Panel cấu hình: chọn model, data, threshold
        ctrl = ttk.LabelFrame(self, text="Cấu hình", padding=10)
        ctrl.pack(fill="x", padx=14, pady=6)
        ctrl.columnconfigure(1, weight=1)

        ttk.Label(ctrl, text="Model:").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        self.model_var = tk.StringVar()
        self.model_cb  = ttk.Combobox(ctrl, textvariable=self.model_var,
                                      values=list(self.models.keys()),
                                      state="readonly", width=65)
        if self.models:
            self.model_cb.current(0)
        self.model_cb.grid(row=0, column=1, padx=6, pady=5, sticky="ew")

        ttk.Label(ctrl, text="Data test:").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        self.data_var = tk.StringVar()
        self.data_cb  = ttk.Combobox(ctrl, textvariable=self.data_var,
                                     values=list(self.datasets.keys()),
                                     state="readonly", width=65)
        if self.datasets:
            self.data_cb.current(0)
        self.data_cb.grid(row=1, column=1, padx=6, pady=5, sticky="ew")

        ttk.Label(ctrl, text="Threshold τ (open-set):").grid(row=2, column=0, sticky="w", padx=6, pady=5)
        self.tau_var = tk.StringVar(value="0.7")
        ttk.Entry(ctrl, textvariable=self.tau_var, width=8).grid(row=2, column=1,
                                                                   padx=6, pady=5, sticky="w")

        btn_row = ttk.Frame(ctrl)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=6)
        self.btn_test = ttk.Button(btn_row, text="   TEST   ", command=self._start_test)
        self.btn_test.pack(side="left", padx=(0, 16))
        self.status_var = tk.StringVar(value="Chưa chạy.")
        ttk.Label(btn_row, textvariable=self.status_var, foreground="#555555").pack(side="left")

        # Panel kết quả: text bên trái, confusion matrix bên phải
        result_pane = ttk.Frame(self)
        result_pane.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        left = ttk.LabelFrame(result_pane, text="Kết quả", padding=6)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.txt = tk.Text(left, font=("Courier New", 10), state="disabled", wrap="none")
        txt_vsb = ttk.Scrollbar(left, orient="vertical", command=self.txt.yview)
        txt_hsb = ttk.Scrollbar(left, orient="horizontal", command=self.txt.xview)
        self.txt.configure(yscrollcommand=txt_vsb.set, xscrollcommand=txt_hsb.set)

        self.txt.grid(row=0, column=0, sticky="nsew")
        txt_vsb.grid(row=0, column=1, sticky="ns")
        txt_hsb.grid(row=1, column=0, sticky="ew")

        right = ttk.LabelFrame(result_pane, text="Confusion Matrix", padding=6)
        right.pack(side="right", fill="both", expand=True)
        self.cm_label = ttk.Label(right, anchor="center")
        self.cm_label.pack(fill="both", expand=True)
        # Resize ảnh khi kéo thay đổi kích thước cửa sổ
        self.cm_label.bind("<Configure>", self._on_cm_resize)

    def _on_cm_resize(self, _event):
        """Resize confusion matrix khi kích thước panel thay đổi."""
        if self._cm_img_orig:
            self._show_cm(self._cm_img_orig)

    def _start_test(self):
        """Validate đầu vào, disable nút, chạy worker thread."""
        if not self.models:
            self._set_text("Không tìm thấy model nào trong saved_models/.")
            return
        if not self.datasets:
            self._set_text("Không tìm thấy dataset nào.")
            return
        self.btn_test.config(state="disabled")
        self.status_var.set("Đang chạy...")
        # Chạy trong thread riêng để không block UI
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        """Worker thread: load model → load data → eval → gửi kết quả về UI thread."""
        try:
            model_key       = self.model_var.get()
            data_key        = self.data_var.get()
            tau             = float(self.tau_var.get())
            data_dir, ds_id = self.datasets[data_key]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model từ checkpoint
            ckpt = torch.load(self.models[model_key], map_location=device, weights_only=False)
            num_classes = ckpt["num_classes"]
            model = CNNGRU(num_classes=num_classes).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # Lấy tên nhãn theo dataset đang test (không phải dataset train)
            cnames = CLASS_NAMES.get(ds_id, [f"Class {i}" for i in range(num_classes)])[:num_classes]

            text, cm_img = self._eval_raw(
                model, data_dir, ds_id, num_classes, cnames, tau, device, model_key, data_key)

            # Gửi kết quả về UI thread (after() thread-safe)
            self.after(0, self._update_ui, text, cm_img, "Hoàn tất.")

        except Exception:
            import traceback
            self.after(0, self._update_ui, f"Lỗi:\n{traceback.format_exc()}", None, "Lỗi.")

    def _eval_raw(self, model, data_dir, ds_id, num_classes,
                  cnames, tau, device, model_key, data_key):
        """Đánh giá trên tín hiệu thô từ Data_test/ (split test 15%).

        Load signals_test.npy + filenames_test.npy → pipeline inference thực tế →
        kết quả đánh giá ở mức tín hiệu (không phải window), kèm bảng chi tiết
        từng tín hiệu với tên file gốc để truy xuất nguồn.
        """
        signals = np.load(os.path.join(data_dir, "signals_test.npy"), allow_pickle=True)
        labels  = np.load(os.path.join(data_dir, "labels_test.npy"))

        # Load filenames nếu có (sau khi chạy lại process_data.py)
        fpath  = os.path.join(data_dir, "filenames_test.npy")
        fnames = list(np.load(fpath)) if os.path.exists(fpath) else None

        # Lọc nhãn ngoài class space
        mask    = labels < num_classes
        n_drop  = int((~mask).sum())
        signals = list(signals[mask])
        labels  = labels[mask]
        if fnames is not None:
            fnames = [fn for fn, m in zip(fnames, mask.tolist()) if m]

        # Resample tín hiệu về đúng sample rate model được train (vd. model merged
        # học ở 12kHz nhưng data mfpt gốc lưu ở 48,828Hz → phải resample trước khi
        # sliding_window, nếu không model sẽ nhận sai domain và đoán bậy).
        model_ds  = model_key.split(" / ", 1)[0]
        model_fs  = FS_BY_DATASET.get(model_ds)
        data_fs   = FS_BY_DATASET.get(ds_id)
        resample_note = ""
        if model_fs and data_fs and model_fs != data_fs:
            signals = [resample_signal(np.asarray(s, dtype=np.float64), data_fs, model_fs)
                       for s in signals]
            resample_note = f" → resample {data_fs}Hz→{model_fs}Hz"

        self.after(0, self.status_var.set,
                   f"Đang inference {len(signals)} tín hiệu (Data_test)...")
        all_preds, all_true, details = _predict_raw(
            model, signals, labels, num_classes, tau, device, filenames=fnames)

        pipeline = f"Data_test [split test 15%]{resample_note} → sliding_window → z-score → vote"
        return self._format_raw_result(
            all_preds, all_true, details, n_drop, cnames, tau, device,
            model_key, data_key, pipeline)

    def _format_raw_result(self, all_preds, all_true, details, n_drop, cnames,
                           tau, device, model_key, data_key, pipeline_info):
        """Tạo text kết quả và confusion matrix cho pipeline tín hiệu thô.

        Accuracy tính theo tín hiệu (sau vote), không phải theo window.
        Classification report và confusion matrix bỏ qua các tín hiệu bị reject (None).
        """
        none_count = int((all_preds == -1).sum())
        total_sig  = len(all_true)
        valid      = all_preds >= 0
        correct    = int((all_preds[valid] == all_true[valid]).sum())
        acc        = correct / total_sig if total_sig > 0 else 0.0

        present = sorted(set(all_true[valid].tolist()) | set(all_preds[valid].tolist()))
        pnames  = [(cnames[i] if i < len(cnames) else f"Class {i}") for i in present]
        report  = classification_report(all_true[valid], all_preds[valid],
                                        labels=present, target_names=pnames, digits=4)
        table   = _make_detail_table(details, cnames)

        text = (
            f"Pipeline : {pipeline_info}\n"
            f"Model    : {model_key}\n"
            f"Data     : {data_key}\n"
            f"Device   : {device}  |  Tín hiệu: {total_sig}"
            + (f"  (bỏ {n_drop} nhãn ngoài class space)" if n_drop else "") + "\n"
            f"{'─'*55}\n"
            f"Accuracy : {acc*100:.4f}%  ({correct}/{total_sig} tín hiệu đúng)\n"
            f"None (τ={tau}) : {none_count} tín hiệu ({none_count/total_sig*100:.1f}%)\n"
            f"{'─'*55}\n"
            f"Classification Report (theo tín hiệu):\n{report}\n"
            f"Chi tiết từng tín hiệu:\n{table}"
        )
        cm_img = _build_cm_image(all_true, all_preds, cnames)
        return text, cm_img

    def _update_ui(self, text, cm_img, status):
        """Cập nhật UI từ main thread (được gọi qua after())."""
        self._set_text(text)
        self._cm_img_orig = cm_img
        if cm_img:
            self._show_cm(cm_img)
        self.status_var.set(status)
        self.btn_test.config(state="normal")

    def _set_text(self, text):
        """Ghi nội dung vào text box (cần enable trước, disable sau)."""
        self.txt.config(state="normal")
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, text)
        self.txt.config(state="disabled")

    def _show_cm(self, img):
        """Resize và hiển thị confusion matrix vừa khít panel."""
        w = max(self.cm_label.winfo_width(),  300)
        h = max(self.cm_label.winfo_height(), 280)
        resized = img.copy()
        resized.thumbnail((w, h), Image.LANCZOS)
        self._cm_photo = ImageTk.PhotoImage(resized)
        self.cm_label.config(image=self._cm_photo)


if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()
