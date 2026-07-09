"""
9 cross-dataset scenarios trên data chuẩn hóa 12 kHz (data_cross/).

S1: Train CWRU,      test CWRU
S2: Train CWRU,      test MFPT
S3: Train CWRU,      test Paderborn
S4: Train MFPT,      test CWRU
S5: Train MFPT,      test MFPT
S6: Train MFPT,      test Paderborn
S7: Train Paderborn, test CWRU
S8: Train Paderborn, test MFPT
S9: Train Paderborn, test Paderborn

Kết quả lưu vào saved_models_cross/{scenario}/
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import classification_report

from train_model import (
    CNNGRU, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    create_dataloaders, evaluate,
    plot_training_history, plot_confusion_matrix,
    plot_lr_schedule, save_results,
)

# CẤU HÌNH
SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))
# processed_data_std/ chứa data đã resample về 12kHz cho cả 3 dataset
# (khác processed_data/ giữ native rate — dùng cho individual training)
DATA_DIR     = os.path.join(SOFTWARE_DIR, "processed_data_std")
SAVE_DIR     = os.path.join(SOFTWARE_DIR, "saved_models")

# 9 kịch bản: (scenario_id, train_dataset, test_dataset)
# S1/S5/S9 là within-dataset (train = test) — baseline so sánh
# S2,S3,S4,S6,S7,S8 là cross-dataset thực sự — đo khả năng generalization
SCENARIOS = [
    ("S1", "cwru",      "cwru"),
    ("S2", "cwru",      "mfpt"),
    ("S3", "cwru",      "paderborn"),
    ("S4", "mfpt",      "cwru"),
    ("S5", "mfpt",      "mfpt"),
    ("S6", "mfpt",      "paderborn"),
    ("S7", "paderborn", "cwru"),
    ("S8", "paderborn", "mfpt"),
    ("S9", "paderborn", "paderborn"),
]

# Tên nhãn theo từng dataset — dùng cho confusion matrix và classification report
# CWRU có 4 lớp (thêm Ball), MFPT và Paderborn chỉ có 3 lớp
CLASS_NAMES = {
    "cwru":      ["Normal", "IR", "OR", "Ball"],
    "mfpt":      ["Baseline", "Inner Race", "Outer Race"],
    "paderborn": ["Normal", "Inner Race", "Outer Race"],
}


# DATA LOADING
def load_data(ds_name: str):
    """Load đầy đủ train/val/test + class_weights từ processed_data_std/{ds_name}/."""
    d = os.path.join(DATA_DIR, ds_name)
    X_tr = np.load(os.path.join(d, "X_train.npy"))
    Y_tr = np.load(os.path.join(d, "Y_train.npy"))
    X_va = np.load(os.path.join(d, "X_val.npy"))
    Y_va = np.load(os.path.join(d, "Y_val.npy"))
    X_te = np.load(os.path.join(d, "X_test.npy"))
    Y_te = np.load(os.path.join(d, "Y_test.npy"))
    cw_path = os.path.join(d, "class_weights.npy")
    cw = np.load(cw_path).astype(np.float32) if os.path.exists(cw_path) else None
    return X_tr, Y_tr, X_va, Y_va, X_te, Y_te, cw


def load_test_only(ds_name: str, num_classes: int):
    """Load chỉ tập test từ dataset khác (dùng cho cross-dataset evaluation).

    Lọc bỏ nhãn >= num_classes vì model train trên dataset nguồn không có lớp đó.
    Ví dụ: model train MFPT (3 lớp) test trên CWRU → bỏ nhãn 3 (Ball) vì MFPT không có.
    """
    d = os.path.join(DATA_DIR, ds_name)
    X_te = np.load(os.path.join(d, "X_test.npy"))
    Y_te = np.load(os.path.join(d, "Y_test.npy"))
    mask = Y_te < num_classes
    n_drop = int((~mask).sum())
    if n_drop:
        print(f"    [filter] Bỏ {n_drop} mẫu nhãn >= {num_classes} "
              f"(ngoài class space của model)")
    return X_te[mask], Y_te[mask]


# TRAINING
def train_scenario(sid: str, train_ds: str, test_ds: str) -> float:
    """Huấn luyện 1 scenario cross-dataset và trả về test accuracy.

    Args:
        sid:      ID scenario (S1..S9)
        train_ds: dataset dùng để train (cwru / mfpt / paderborn)
        test_ds:  dataset dùng để test  (cwru / mfpt / paderborn)

    Returns:
        test accuracy (float) của best model trên tập test
    """
    label    = f"{sid}_{train_ds}_to_{test_ds}"
    save_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  {label.upper()}  |  device={device}")
    print(f"{'='*65}")

    # Load train/val từ dataset nguồn; test từ dataset đích
    # Nếu train_ds == test_ds (S1/S5/S9): dùng luôn tập test của dataset nguồn
    # Nếu khác nhau: load tập test riêng của dataset đích và lọc nhãn thừa
    X_tr, Y_tr, X_va, Y_va, X_te_own, Y_te_own, cw = load_data(train_ds)
    num_classes = int(Y_tr.max()) + 1

    X_te = X_te_own if test_ds == train_ds else None
    Y_te = Y_te_own if test_ds == train_ds else None
    if X_te is None:
        X_te, Y_te = load_test_only(test_ds, num_classes)

    print(f"  Train {X_tr.shape}  Val {X_va.shape}  Test {X_te.shape}")
    print(f"  num_classes={num_classes}  "
          f"train_labels={sorted(np.unique(Y_tr).tolist())}  "
          f"test_labels={sorted(np.unique(Y_te).tolist())}")

    train_loader, val_loader, test_loader = create_dataloaders(
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te)

    # Khởi tạo model với số lớp của dataset nguồn (train_ds)
    model = CNNGRU(num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # Loss: dùng class_weights của dataset nguồn khi train để xử lý mất cân bằng lớp
    # Val/test dùng unweighted để đánh giá khách quan trên phân phối thực
    if cw is not None:
        criterion_train = nn.CrossEntropyLoss(
            weight=torch.tensor(cw, dtype=torch.float32, device=device))
        print(f"  class_weights: {cw.tolist()}")
    else:
        criterion_train = nn.CrossEntropyLoss()
    criterion_eval = nn.CrossEntropyLoss()

    # Dùng CosineAnnealingLR thay vì ReduceLROnPlateau (mặc định train_model.py)
    # Cosine phù hợp hơn cho cross-dataset: LR giảm đều, không phụ thuộc vào val_loss
    # của dataset nguồn (vốn không đại diện cho dataset đích)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Lưu toàn bộ metrics theo từng epoch để vẽ biểu đồ sau
    history = {k: [] for k in [
        "train_loss", "val_loss", "train_acc", "val_acc",
        "train_loss_std", "val_loss_std", "train_acc_std", "val_acc_std", "lr",
    ]}
    best_val_loss = float("inf")
    best_state    = None  # lưu weights của epoch có val_loss thấp nhất

    print(f"\n  Epochs: {NUM_EPOCHS}  Batch: {BATCH_SIZE}  LR: {LEARNING_RATE}")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train phase
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        b_losses, b_accs = [], []

        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion_train(logits, Yb)
            loss.backward()
            optimizer.step()

            bl = loss.item()
            b_losses.append(bl)
            t_loss += bl * Xb.size(0)
            preds   = logits.argmax(1)
            b_accs.append((preds == Yb).float().mean().item())
            t_correct += (preds == Yb).sum().item()
            t_total   += Xb.size(0)

        t_loss /= t_total
        t_acc   = t_correct / t_total

        # Validation dùng unweighted loss để đánh giá khách quan
        v_loss, v_acc, v_loss_std, v_acc_std = evaluate(
            model, val_loader, device, criterion_eval)

        # CosineAnnealingLR step per epoch (không cần truyền val_loss)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["train_loss_std"].append(float(np.std(b_losses)))
        history["val_loss_std"].append(v_loss_std)
        history["train_acc_std"].append(float(np.std(b_accs)))
        history["val_acc_std"].append(v_acc_std)
        history["lr"].append(lr)

        # Lưu weights tốt nhất dựa trên val_loss (dataset nguồn)
        # best_state lưu trên CPU để tiết kiệm VRAM khi train dài
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"train loss={t_loss:.4f} acc={t_acc:.4f} | "
                  f"val loss={v_loss:.4f} acc={v_acc:.4f} | "
                  f"lr={lr:.2e}")

    # Rollback về epoch tốt nhất trước khi đánh giá cuối
    model.load_state_dict(best_state)
    print(f"\n  Best val_loss={best_val_loss:.6f} → loaded")

    # Đánh giá cuối trên cả 3 split bằng unweighted loss
    tr_loss, tr_acc, tr_ls, tr_as = evaluate(model, train_loader, device, criterion_eval)
    va_loss, va_acc, va_ls, va_as = evaluate(model, val_loader,   device, criterion_eval)
    te_loss, te_acc, te_ls, te_as = evaluate(model, test_loader,  device, criterion_eval)

    print(f"  Train : loss={tr_loss:.4f} acc={tr_acc:.4f} ({tr_acc*100:.2f}%)")
    print(f"  Val   : loss={va_loss:.4f} acc={va_acc:.4f} ({va_acc*100:.2f}%)")
    print(f"  Test  : loss={te_loss:.4f} acc={te_acc:.4f} ({te_acc*100:.2f}%)")

    # Lấy toàn bộ prediction trên tập test để vẽ confusion matrix
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for Xb, Yb in test_loader:
            preds = model(Xb.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(Yb.numpy())
    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    # Lấy tên nhãn của train_ds (vì model được định nghĩa theo không gian lớp của train_ds)
    # Chỉ lấy các nhãn thực sự xuất hiện trong tập test để tránh cột trống trong report
    present_labels = sorted(set(all_true.tolist()) | set(all_preds.tolist()))
    all_cnames     = CLASS_NAMES[train_ds]
    cnames         = [all_cnames[i] for i in present_labels if i < len(all_cnames)]

    print(f"\n  Classification Report (test={test_ds.upper()}):")
    print(classification_report(all_true, all_preds,
                                labels=present_labels,
                                target_names=cnames, digits=4))

    # Lưu model weights + toàn bộ biểu đồ và metrics
    torch.save(
        {"model_state_dict": model.state_dict(), "num_classes": num_classes,
         "test_accuracy": te_acc, "train_ds": train_ds, "test_ds": test_ds},
        os.path.join(save_dir, f"model_{label}.pth"),
    )
    plot_training_history(history, save_dir)
    plot_confusion_matrix(all_true, all_preds, all_cnames, save_dir)
    plot_lr_schedule(history, save_dir)
    save_results(
        history, all_true, all_preds, cnames,
        te_acc, te_loss, te_as, te_ls,
        tr_acc, tr_loss, tr_as, tr_ls,
        va_acc, va_loss, va_as, va_ls,
        save_dir,
    )
    print(f"  Đã lưu → {save_dir}")
    return float(te_acc)


# MAIN
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train 9 cross-dataset scenarios trên data_cross/")
    parser.add_argument(
        "--scenario", default="all",
        help="Scenario cần chạy: S1..S9 hoặc 'all' (mặc định: all)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Danh sách scenario bỏ qua, vd: --skip S1 S2",
    )
    args = parser.parse_args()

    run_all  = args.scenario.lower() == "all"
    skip_set = {s.upper() for s in (args.skip or [])}
    results  = {}

    for sid, train_ds, test_ds in SCENARIOS:
        if sid in skip_set:
            print(f"  [SKIP] {sid} ({train_ds} → {test_ds})")
            continue
        if not run_all and args.scenario.upper() != sid:
            continue
        acc = train_scenario(sid, train_ds, test_ds)
        results[sid] = (train_ds, test_ds, acc)

    # In bảng tổng kết tất cả scenario đã chạy
    if results:
        print(f"\n{'='*65}")
        print("  KẾT QUẢ CÁC SCENARIO")
        print(f"{'='*65}")
        for sid, train_ds, test_ds in SCENARIOS:
            if sid not in results:
                continue
            _, _, acc = results[sid]
            print(f"  {sid}: train {train_ds.upper():<10} test {test_ds.upper():<10} "
                  f"→ Test Accuracy = {acc*100:.2f}%")
        print(f"{'='*65}")
