"""
Công cụ hỗ trợ: kiểm tra checkpoint, xem file dữ liệu, tạo cross-checkpoints.

Cách dùng:
    python tools.py check                          # kiểm tra tất cả checkpoint
    python tools.py inspect <file.npy|file.pth>    # xem nội dung file
    python tools.py make-checkpoints               # tạo checkpoint cho cross jobs
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch

SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))

CROSS_JOBS = [
    ("cwru",      "mfpt"),
    ("mfpt",      "paderborn"),
    ("paderborn", "cwru"),
]


# CHECK PROGRESS
def cmd_check(_args):
    """Kiểm tra trạng thái checkpoint của tất cả các job đã train."""
    saved_dir = os.path.join(SOFTWARE_DIR, "saved_models")
    if not os.path.isdir(saved_dir):
        print("Chưa có thư mục saved_models/.")
        return

    runs = sorted(os.listdir(saved_dir))
    if not runs:
        print("Chưa có checkpoint nào.")
        return

    print(f"\n{'=' * 85}")
    print(f"  {'RUN':<28} {'EPOCH':>6}  {'TRAIN ACC':>9}  {'VAL ACC':>9}  {'BEST VAL LOSS':>13}  {'LR':>8}  UPDATED")
    print(f"  {'-' * 83}")

    for run in runs:
        ckpt_path = os.path.join(saved_dir, run, "checkpoint_latest.pth")
        if not os.path.exists(ckpt_path):
            print(f"  {run:<28}  (không có checkpoint)")
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ep      = ckpt["epoch"]
            h       = ckpt["history"]
            bvl     = ckpt["best_val_loss"]
            ta      = h["train_acc"][-1]
            va      = h["val_acc"][-1]
            lr      = h["lr"][-1]
            mtime   = os.path.getmtime(ckpt_path)
            updated = datetime.datetime.fromtimestamp(mtime).strftime("%H:%M %d/%m")
            print(f"  {run:<28} {ep:>6}  {ta:>9.4f}  {va:>9.4f}  {bvl:>13.6f}  {lr:>8.2e}  {updated}")
        except Exception as e:
            print(f"  {run:<28}  [Lỗi đọc: {e}]")

    print(f"{'=' * 85}\n")


# INSPECT FILE
def _inspect_npy(path):
    data = np.load(path)
    print(f"  Shape : {data.shape}")
    print(f"  Dtype : {data.dtype}")
    print(f"  Min   : {data.min():.6f}")
    print(f"  Max   : {data.max():.6f}")
    print(f"  Mean  : {data.mean():.6f}")
    print(f"  Std   : {data.std():.6f}")
    if data.ndim == 1:
        unique, counts = np.unique(data, return_counts=True)
        print(f"\n  Phân bố nhãn:")
        for u, c in zip(unique, counts):
            print(f"    Label {int(u)}: {c} mẫu ({c/len(data)*100:.1f}%)")
    elif data.ndim == 2:
        print(f"\n  5 mẫu đầu (10 giá trị đầu mỗi mẫu):")
        for i in range(min(5, data.shape[0])):
            print(f"    [{i}]: {data[i, :10]} ...")


def _inspect_pth(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Keys: {list(ckpt.keys())}")
    for key, val in ckpt.items():
        if isinstance(val, dict):
            print(f"    '{key}': dict ({len(val)} keys)")
        elif isinstance(val, torch.Tensor):
            print(f"    '{key}': Tensor {val.shape}")
        else:
            print(f"    '{key}': {val}")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        total = sum(t.numel() for t in state.values())
        print(f"\n  Layers ({total:,} params tổng):")
        for name, tensor in state.items():
            print(f"    {name:<52} {str(tensor.shape):<25} {tensor.numel():,}")


def cmd_inspect(args):
    path = args.file
    if not os.path.exists(path):
        print(f"Không tìm thấy: {path}")
        return
    ext = os.path.splitext(path)[1].lower()
    print(f"\n{'=' * 60}")
    print(f"  {os.path.basename(path)}")
    print(f"  {path}")
    print(f"{'=' * 60}")
    if ext == ".npy":
        _inspect_npy(path)
    elif ext == ".pth":
        _inspect_pth(path)
    else:
        print(f"Không hỗ trợ định dạng '{ext}'. Chỉ hỗ trợ .npy và .pth.")
    print()


# MAKE CROSS CHECKPOINTS
def cmd_make_checkpoints(_args):
    """Tạo checkpoint_latest.pth cho các cross jobs từ file .pth và .json đã có.

    Dùng khi muốn resume cross jobs từ epoch > 100.
    """
    from train_model import CNNGRU

    for train_ds, test_ds in CROSS_JOBS:
        run_label = f"{train_ds}_to_{test_ds}"
        save_dir  = os.path.join(SOFTWARE_DIR, "saved_models", run_label)
        json_path  = os.path.join(save_dir, "training_results.json")
        model_path = os.path.join(save_dir, f"cnn_gru_{run_label}.pth")
        ckpt_path  = os.path.join(save_dir, "checkpoint_latest.pth")

        if not os.path.exists(json_path) or not os.path.exists(model_path):
            print(f"[SKIP] {run_label}: thiếu file")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        total_epochs  = results["total_epochs"]
        best_val_loss = results["best_val_loss"]
        final_lr      = results["history"]["learning_rate"][-1]

        history = {
            "train_loss":     results["history"]["train_loss"],
            "val_loss":       results["history"]["val_loss"],
            "train_acc":      results["history"]["train_acc"],
            "val_acc":        results["history"]["val_acc"],
            "train_loss_std": results["history"]["train_loss_std"],
            "val_loss_std":   results["history"]["val_loss_std"],
            "train_acc_std":  results["history"]["train_acc_std"],
            "val_acc_std":    results["history"]["val_acc_std"],
            "lr":             results["history"]["learning_rate"],
        }

        saved = torch.load(model_path, map_location="cpu", weights_only=False)
        model = CNNGRU(num_classes=saved["num_classes"])
        model.load_state_dict(saved["model_state_dict"])

        optimizer = torch.optim.Adam(model.parameters(), lr=final_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        torch.save({
            "epoch":                total_epochs,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss":        best_val_loss,
            "best_model_state":     model.state_dict(),
            "history":              history,
        }, ckpt_path)

        print(f"[OK] {run_label:<30} epoch={total_epochs}  LR={final_lr:.2e}  best_val={best_val_loss:.6f}")
        print(f"     → {ckpt_path}")

    print("\nXong! Chạy: python run.py --jobs cross --epochs 200 --resume")


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Công cụ hỗ trợ: checkpoint / inspect / make-checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python tools.py check
  python tools.py inspect processed_data/cwru/X_train.npy
  python tools.py inspect saved_models/cwru/cnn_gru_cwru.pth
  python tools.py make-checkpoints
        """,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="Kiểm tra trạng thái tất cả checkpoint")

    p_inspect = sub.add_parser("inspect", help="Xem nội dung file .npy hoặc .pth")
    p_inspect.add_argument("file", help="Đường dẫn tới file .npy hoặc .pth")

    sub.add_parser("make-checkpoints", help="Tạo checkpoint_latest.pth cho cross jobs")

    args = parser.parse_args()
    {"check": cmd_check, "inspect": cmd_inspect, "make-checkpoints": cmd_make_checkpoints}[args.cmd](args)
