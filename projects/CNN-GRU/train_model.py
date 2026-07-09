"""
Train Model: CNN-GRU cho bearing fault diagnosis.

Kiến trúc theo Train_model.md (5 bước):
    Bước 1: Concatenate 2 nhánh → sequence (B, 32, 256), không cần reshape
    Bước 2: GRU layer — unroll qua T=32 timesteps
    Bước 3: Mean pool theo T + FC + Dropout + ReLU
    Bước 4: Softmax + Class-weighted Cross-Entropy
    Bước 5: Lưu model

Open-set: ở inference, p_max < τ → predict None (xem inference.py).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from extract_data import FeatureExtractor, FEATURE_DIM, SEQUENCE_LEN


# CẤU HÌNH
SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
LOSS_THRESHOLD = 3e-6
DROPOUT_RATE = 0.3     # Dropout probability (p)

# GRU config
GRU_HIDDEN_SIZE = 128
GRU_NUM_LAYERS = 1

# FC config
FC_HIDDEN_SIZE = 64


# MÔ HÌNH CNN-GRU
class CNNGRU(nn.Module):
    """
    Mô hình CNN-GRU cho phân loại lỗi vòng bi.

    Pipeline:
        Input (B, 2048)
            → FeatureExtractor (CNN1D + CNN2D) → (B, 32, 256)
            → Bước 2: GRU unroll qua T=32 → (B, 32, hidden)
            → Bước 3: Mean pool theo T → (B, hidden) → FC + Dropout + ReLU
            → Bước 4: Softmax (trong CrossEntropyLoss)
            → Output: logits (B, num_classes)
    """

    def __init__(self, num_classes):
        super(CNNGRU, self).__init__()

        concat_dim = FEATURE_DIM * 2  # 128 * 2 = 256

        # Bước 1: Feature Extractor đã trả về sequence (B, 32, 256)
        self.feature_extractor = FeatureExtractor()

        # Bước 2: GRU layer — input là sequence thực sự (T=32)
        # Γ_u = σ(W_u[c^(t-1), x^(t)] + b_u)
        # Γ_r = σ(W_r[c^(t-1), x^(t)] + b_r)
        # c̃^(t) = tanh(W_c[Γ_r * c^(t-1), x^(t)] + b_c)
        # c^(t) = (1 - Γ_u) * c^(t-1) + Γ_u * c̃^(t)
        self.gru = nn.GRU(
            input_size=concat_dim,
            hidden_size=GRU_HIDDEN_SIZE,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True,
        )

        # Bước 3: FC + Dropout + ReLU
        self.fc1 = nn.Linear(GRU_HIDDEN_SIZE, FC_HIDDEN_SIZE)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.relu = nn.ReLU()

        # Bước 4: Output layer (logits → Softmax trong CrossEntropyLoss)
        self.fc_out = nn.Linear(FC_HIDDEN_SIZE, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, 2048) - tín hiệu rung thô
        Returns:
            logits: (B, num_classes) - raw scores (Softmax áp dụng trong loss)
        """
        # Bước 1: CNN1D + CNN2D → sequence (B, 32, 256)
        z = self.feature_extractor(x)

        # Bước 2: GRU unroll qua T=32 timesteps thực sự
        gru_out, _ = self.gru(z)              # (B, 32, hidden_size)

        # Bước 3a: Mean pool theo trục thời gian
        h = gru_out.mean(dim=1)               # (B, hidden_size)

        # Bước 3b: FC + Dropout + ReLU
        h = self.fc1(h)                       # (B, 64)
        h = self.dropout(h)
        h = self.relu(h)

        # Bước 4: Output logits
        logits = self.fc_out(h)               # (B, num_classes)

        return logits


# HÀM HỖ TRỢ
def load_dataset(dataset_name):
    """Load processed data + class_weights từ thư mục processed_data/."""
    data_dir = os.path.join(SOFTWARE_DIR, "processed_data", dataset_name)

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    Y_test = np.load(os.path.join(data_dir, "Y_test.npy"))

    # Class weights (sqrt inverse frequency) — tính sẵn ở bước Read_data
    class_weights_path = os.path.join(data_dir, "class_weights.npy")
    if os.path.exists(class_weights_path):
        class_weights = np.load(class_weights_path).astype(np.float32)
    else:
        class_weights = None
        print(f"  [Warning] Không tìm thấy {class_weights_path} — sẽ dùng CE không weight")

    print(f"Dataset: {dataset_name.upper()}")
    print(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
    print(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")
    print(f"  Classes: {np.unique(Y_train)}")
    if class_weights is not None:
        print(f"  Class weights (sqrt): {class_weights.tolist()}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, class_weights


def load_dataset_full(dataset_name):
    """Load toàn bộ data (train+val+test gộp lại) — không có split."""
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te, class_weights = load_dataset(dataset_name)
    X_all = np.concatenate([X_tr, X_va, X_te], axis=0)
    Y_all = np.concatenate([Y_tr, Y_va, Y_te], axis=0)
    print(f"  [Full data] Gộp train+val+test: X={X_all.shape}, Y={Y_all.shape}")
    return X_all, Y_all, class_weights


def load_test_split(dataset_name, num_classes):
    """Load test split từ dataset khác (dùng cho cross-dataset evaluation).
    Lọc bỏ các mẫu có nhãn >= num_classes (ngoài không gian lớp của model).
    """
    data_dir = os.path.join(SOFTWARE_DIR, "processed_data", dataset_name)
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    Y_test = np.load(os.path.join(data_dir, "Y_test.npy"))

    print(f"\n  [Cross-test] Load test từ {dataset_name.upper()}: {X_test.shape}")
    mask = Y_test < num_classes
    n_filtered = (~mask).sum()
    if n_filtered > 0:
        print(f"  [Cross-test] Lọc {n_filtered} mẫu có nhãn >= {num_classes} (ngoài class space)")
        X_test, Y_test = X_test[mask], Y_test[mask]
    print(f"  [Cross-test] Sau lọc: {X_test.shape}, classes={np.unique(Y_test).tolist()}")
    return X_test, Y_test


def create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """Tạo PyTorch DataLoader từ numpy arrays."""
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)   # shuffle để tránh model học thứ tự batch
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)      # không shuffle: kết quả đánh giá ổn định
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate(model, dataloader, device, criterion):
    """Đánh giá model — trả về (avg_loss, accuracy, loss_std, acc_std) tính theo batch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_losses = []
    batch_accs = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, Y_batch)

            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            # Nhân theo số sample để tính trung bình có trọng số (weighted mean)
            # tránh sai lệch khi batch cuối nhỏ hơn BATCH_SIZE
            total_loss += batch_loss * X_batch.size(0)

            preds = logits.argmax(dim=1)
            batch_accs.append((preds == Y_batch).float().mean().item())
            correct += (preds == Y_batch).sum().item()
            total += X_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    # std tính theo batch (không phải sample) — dùng để vẽ error bar, phản ánh độ ổn định
    loss_std = float(np.std(batch_losses))
    acc_std = float(np.std(batch_accs))
    return avg_loss, accuracy, loss_std, acc_std


# VISUALIZATION & TRACKING
def plot_training_history(history, save_dir):
    """Vẽ biểu đồ Loss và Accuracy qua các epoch."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-o', markersize=3, label='Val Loss')
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = min(history['val_loss'])
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss Curves (best val_loss={best_val:.6g})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history['train_acc'], 'b-o', markersize=3, label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-o', markersize=3, label='Val Acc')
    best_acc_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    axes[1].axvline(x=best_acc_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {best_acc_epoch})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Accuracy Curves (best val_acc={best_acc:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Vẽ Confusion Matrix heatmap."""
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names  = [class_names[i] for i in present_labels if i < len(class_names)]
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    n = len(present_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='%')

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=present_names,
        yticklabels=present_names,
        xlabel='Predicted',
        ylabel='True',
        title='Confusion Matrix (%)',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Ghi số vào từng ô
    for i in range(n):
        for j in range(n):
            color = 'white' if cm_percent[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                    ha='center', va='center', color=color, fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] {path}")


def save_results(history, y_true, y_pred, class_names,
                 test_acc, test_loss, test_acc_std, test_loss_std,
                 train_acc, train_loss, train_acc_std, train_loss_std,
                 val_acc, val_loss, val_acc_std, val_loss_std,
                 save_dir):
    """Lưu toàn bộ kết quả ra JSON + classification report txt."""
    # Classification report
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names  = [class_names[i] for i in present_labels if i < len(class_names)]
    report_str = classification_report(y_true, y_pred,
                                       labels=present_labels, target_names=present_names, digits=4)
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_str)
    print(f"  [Saved] {report_path}")

    # Training history JSON
    results = {
        'train': {'accuracy': train_acc, 'loss': train_loss, 'accuracy_std': train_acc_std, 'loss_std': train_loss_std},
        'val':   {'accuracy': val_acc,   'loss': val_loss,   'accuracy_std': val_acc_std,   'loss_std': val_loss_std},
        'test':  {'accuracy': test_acc,  'loss': test_loss,  'accuracy_std': test_acc_std,  'loss_std': test_loss_std},
        'best_val_loss': float(min(history['val_loss'])),
        'best_val_acc': float(max(history['val_acc'])),
        'total_epochs': len(history['train_loss']),
        'history': {
            'train_loss':     [float(x) for x in history['train_loss']],
            'val_loss':       [float(x) for x in history['val_loss']],
            'train_acc':      [float(x) for x in history['train_acc']],
            'val_acc':        [float(x) for x in history['val_acc']],
            'train_loss_std': [float(x) for x in history['train_loss_std']],
            'val_loss_std':   [float(x) for x in history['val_loss_std']],
            'train_acc_std':  [float(x) for x in history['train_acc_std']],
            'val_acc_std':    [float(x) for x in history['val_acc_std']],
            'learning_rate':  [float(x) for x in history['lr']],
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred,
                                                        labels=present_labels,
                                                        target_names=present_names,
                                                        output_dict=True),
    }
    json_path = os.path.join(save_dir, 'training_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [Saved] {json_path}")


def plot_lr_schedule(history, save_dir):
    """Vẽ biểu đồ learning rate qua các epoch."""
    epochs = range(1, len(history['lr']) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history['lr'], 'g-o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'lr_schedule.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] {path}")



# TRAINING LOOP
def train(dataset_name="cwru", lr=LEARNING_RATE, epochs=NUM_EPOCHS, scheduler_name='plateau', eta_min=1e-6, resume=False, test_dataset_name=None, use_all_data=False):
    """
    Huấn luyện mô hình CNN-GRU.
    Dừng khi train_loss < LOSS_THRESHOLD hoặc đạt epochs tối đa.
    Lưu model có val_loss nhỏ nhất trong toàn bộ quá trình train.
    test_dataset_name: nếu khác None, dùng tập test từ dataset khác (cross-dataset).
    use_all_data: nếu True, gộp train+val+test thành 1 tập, train trên toàn bộ data.
    """
    is_cross = test_dataset_name is not None and test_dataset_name != dataset_name
    run_label = f"{dataset_name}_to_{test_dataset_name}" if is_cross else dataset_name

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data + class weights
    if use_all_data:
        X_all, Y_all, class_weights = load_dataset_full(dataset_name)
        num_classes = int(Y_all.max() + 1)
        print(f"Số lớp (C): {num_classes}")
        train_loader, val_loader, test_loader = create_dataloaders(
            X_all, Y_all, X_all, Y_all, X_all, Y_all
        )
    else:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, class_weights = load_dataset(dataset_name)
        num_classes = int(max(np.unique(Y_train).max() + 1, len(class_weights) if class_weights is not None else 0))
        print(f"Số lớp (C): {num_classes}")

        # Cross-dataset: thay tập test bằng dataset khác
        if is_cross:
            X_test, Y_test = load_test_split(test_dataset_name, num_classes)

        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, Y_train, X_val, Y_val, X_test, Y_test
        )

    # Khởi tạo model
    model = CNNGRU(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Bước 4: Loss function
    # Class-weighted Categorical Cross-Entropy: L = -Σ w_k · y_k log(ŷ_k)
    # w_k = sqrt(N / (C * n_k)) đã tính sẵn ở bước Read_data
    # Train dùng weighted, val/test dùng unweighted (đánh giá unbiased trên phân phối gốc)
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion_train = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"  Loss: weighted CrossEntropy (sqrt inverse frequency)")
    else:
        criterion_train = nn.CrossEntropyLoss()
        print(f"  Loss: standard CrossEntropy (không có class_weights)")
    criterion_eval = nn.CrossEntropyLoss()  # val/test không weight

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler:
    #   plateau   — giảm LR x0.5 khi val_loss không cải thiện sau 5 epoch (mặc định, an toàn)
    #   cosine    — LR giảm dần theo hình cos từ lr về eta_min (mượt, phù hợp fine-tune)
    #   onecycle  — LR tăng rồi giảm theo 1 chu kỳ, step mỗi batch (hội tụ nhanh)
    scheduler = None
    if scheduler_name == 'onecycle':
        # OneCycleLR cần biết tổng số bước (epochs × số batch/epoch)
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
        scheduler_type = 'OneCycleLR'
    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        scheduler_type = 'CosineAnnealingLR'
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        scheduler_type = 'ReduceLROnPlateau'

    # Khởi tạo trạng thái training
    best_val_loss = float('inf')
    best_model_state = None
    start_epoch = 1
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_loss_std': [], 'val_loss_std': [],
        'train_acc_std': [], 'val_acc_std': [],
        'lr': [],
    }

    # Resume từ checkpoint nếu có
    save_dir = os.path.join(SOFTWARE_DIR, "saved_models", run_label)
    checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")

    if resume and os.path.exists(checkpoint_path):
        print(f"\n  [Resume] Load checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Khôi phục đầy đủ: weights, optimizer state (momentum/Adam), scheduler state, history
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict') and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch      = ckpt['epoch'] + 1
        best_val_loss    = ckpt['best_val_loss']
        best_model_state = ckpt['best_model_state']
        history          = ckpt['history']
        # Pad các key mới thêm về sau để tương thích checkpoint cũ
        n = len(history['train_loss'])
        for key in ('train_loss_std', 'val_loss_std', 'train_acc_std', 'val_acc_std'):
            if key not in history:
                history[key] = [0.0] * n
        if 'lr' not in history:
            history['lr'] = [0.0] * n
        print(f"  [Resume] Tiếp tục từ epoch {start_epoch} (best val_loss={best_val_loss:.4f})")
    elif resume:
        print(f"  [Resume] Không tìm thấy checkpoint tại {checkpoint_path} — train từ đầu")

    # Training
    print(f"\n{'=' * 65}")
    print(f"  BẮT ĐẦU HUẤN LUYỆN - {run_label.upper()}")
    print(f"  Epochs: {start_epoch}→{epochs}, Batch size: {BATCH_SIZE}, LR: {lr}")
    print(f"  Scheduler: {scheduler_type}, eta_min: {eta_min}")
    print(f"  Mục tiêu train_loss: {LOSS_THRESHOLD:.2e}")
    print(f"{'=' * 65}\n")

    for epoch in range(start_epoch, epochs + 1):
        # Train phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_train_losses = []
        batch_train_accs = []

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)

            # Bước 4: Class-weighted Cross-Entropy loss (bao gồm Softmax)
            loss = criterion_train(logits, Y_batch)

            loss.backward()
            optimizer.step()

            # If using OneCycleLR we must step per batch
            if scheduler_name == 'onecycle':
                scheduler.step()

            batch_loss = loss.item()
            batch_train_losses.append(batch_loss)
            train_loss += batch_loss * X_batch.size(0)
            preds = logits.argmax(dim=1)
            batch_train_accs.append((preds == Y_batch).float().mean().item())
            train_correct += (preds == Y_batch).sum().item()
            train_total += X_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        train_loss_std = float(np.std(batch_train_losses))
        train_acc_std = float(np.std(batch_train_accs))

        # Validation phase (không weight)
        val_loss, val_acc, val_loss_std, val_acc_std = evaluate(model, val_loader, device, criterion_eval)

        # OneCycleLR đã được step mỗi batch bên trong vòng lặp trên
        # Plateau và Cosine thì step mỗi epoch (Plateau cần truyền val_loss để quyết định giảm LR)
        if scheduler_name == 'onecycle':
            pass
        elif scheduler_name == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # Track history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss_std'].append(train_loss_std)
        history['val_loss_std'].append(val_loss_std)
        history['train_acc_std'].append(train_acc_std)
        history['val_acc_std'].append(val_acc_std)
        history['lr'].append(current_lr)

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f}±{train_loss_std:.6f}, Acc: {train_acc:.4f}±{train_acc_std:.4f} | "
                f"Val Loss: {val_loss:.6f}±{val_loss_std:.6f}, Acc: {val_acc:.4f}±{val_acc_std:.4f} | "
                f"LR: {current_lr:.2e}"
            )

        # Tiêu chí chọn best: val_loss thấp nhất (không phải val_acc)
        # val_loss phản ánh độ tin cậy xác suất, val_acc chỉ đếm đúng/sai
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Ghi đè checkpoint_latest.pth sau mỗi epoch để có thể resume nếu bị ngắt giữa chừng
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'best_model_state': best_model_state,
            'history': history,
        }, checkpoint_path)

    # Sau khi train xong, rollback về checkpoint tốt nhất thay vì dùng weights cuối cùng
    # (epoch cuối chưa chắc tốt hơn — có thể đã overfit)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Đã load lại model tốt nhất (val_loss={best_val_loss:.4f})")

    # Đánh giá cuối trên train / val / test
    print(f"\n{'=' * 65}")
    print("  ĐÁNH GIÁ CUỐI (best model — min val_loss toàn bộ epoch)")
    print(f"{'=' * 65}")
    train_loss_final, train_acc_final, train_loss_std_final, train_acc_std_final = evaluate(
        model, train_loader, device, criterion_eval)
    val_loss_final, val_acc_final, val_loss_std_final, val_acc_std_final = evaluate(
        model, val_loader, device, criterion_eval)
    test_loss, test_acc, test_loss_std, test_acc_std = evaluate(
        model, test_loader, device, criterion_eval)
    print(f"  Train: Loss={train_loss_final:.6f}±{train_loss_std_final:.6f} | Acc={train_acc_final:.4f}±{train_acc_std_final:.4f} ({train_acc_final*100:.2f}%)")
    print(f"  Val:   Loss={val_loss_final:.6f}±{val_loss_std_final:.6f} | Acc={val_acc_final:.4f}±{val_acc_std_final:.4f} ({val_acc_final*100:.2f}%)")
    print(f"  Test:  Loss={test_loss:.6f}±{test_loss_std:.6f} | Acc={test_acc:.4f}±{test_acc_std:.4f} ({test_acc*100:.2f}%)")

    # Predictions trên tập test
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(Y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Tên nhãn dùng cho confusion matrix và classification report
    # Cắt theo num_classes thực tế: cross-dataset có thể ít lớp hơn tên được định nghĩa sẵn
    if dataset_name == "cwru":
        class_names = ['Normal', 'IR (Inner)', 'OR (Outer)', 'B (Ball)']
    elif dataset_name == "mfpt":
        class_names = ['Baseline', 'Inner Race', 'Outer Race']
    elif dataset_name == "paderborn":
        class_names = ['Normal', 'Inner Race', 'Outer Race']
    elif dataset_name == "merged":
        class_names = ['Normal', 'IR', 'OR', 'Ball']
    else:
        class_names = [f'Class {i}' for i in range(num_classes)]
    class_names = class_names[:num_classes]

    # Print classification report
    present_labels = sorted(set(all_labels.tolist()) | set(all_preds.tolist()))
    present_names  = [class_names[i] for i in present_labels if i < len(class_names)]
    print(f"\n  Classification Report:")
    print(classification_report(all_labels, all_preds,
                                labels=present_labels, target_names=present_names, digits=4))

    # Bước 6: Lưu model + results
    save_dir = os.path.join(SOFTWARE_DIR, "saved_models", run_label)
    os.makedirs(save_dir, exist_ok=True)

    # Lưu model weights
    model_path = os.path.join(save_dir, f"cnn_gru_{run_label}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': lr,
            'gru_hidden_size': GRU_HIDDEN_SIZE,
            'fc_hidden_size': FC_HIDDEN_SIZE,
            'dropout_rate': DROPOUT_RATE,
            'feature_dim': FEATURE_DIM,
        }
    }, model_path)
    print(f"\n  Model đã lưu tại: {model_path}")
    if is_cross:
        print(f"  [Cross] Train: {dataset_name.upper()} → Test: {test_dataset_name.upper()}")

    # Vẽ biểu đồ & lưu kết quả
    print(f"\n{'─' * 65}")
    print("  LƯU KẾT QUẢ TRACKING")
    print(f"{'─' * 65}")

    plot_training_history(history, save_dir)
    plot_confusion_matrix(all_labels, all_preds, class_names, save_dir)
    plot_lr_schedule(history, save_dir)
    save_results(history, all_labels, all_preds, class_names,
                 test_acc, test_loss, test_acc_std, test_loss_std,
                 train_acc_final, train_loss_final, train_acc_std_final, train_loss_std_final,
                 val_acc_final, val_loss_final, val_acc_std_final, val_loss_std_final,
                 save_dir)

    print(f"\n  Tất cả kết quả đã lưu tại: {save_dir}")
    print(f"    - training_curves.png      (Loss & Accuracy curves theo epoch)")
    print(f"    - confusion_matrix.png     (Confusion Matrix heatmap)")
    print(f"    - lr_schedule.png          (Learning Rate schedule)")
    print(f"    - classification_report.txt (Precision/Recall/F1)")
    print(f"    - training_results.json    (Toàn bộ history + metrics)")
    print(f"    - cnn_gru_{run_label}.pth (Model weights)")

    print(f"\n{'=' * 65}")
    print("  HOÀN TẤT")
    print(f"{'=' * 65}")

    return model, test_acc


# MAIN
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN-GRU model")
    parser.add_argument(
        "--dataset", type=str, default="merged",
        choices=["cwru", "mfpt", "paderborn", "merged"],
        help="Dataset để train (default: merged — unified model trên 3 dataset gộp)"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs (default: 200)")
    parser.add_argument("--scheduler", type=str, default='plateau', choices=['plateau','onecycle','cosine'], help="LR scheduler: plateau, onecycle, or cosine (default: plateau)")
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Minimum LR for cosine annealing (default: 1e-6)")
    parser.add_argument("--test-dataset", type=str, default=None,
                        choices=["cwru", "mfpt", "paderborn", "merged"],
                        help="Dataset dùng để test (cross-dataset). Mặc định = --dataset")
    parser.add_argument("--resume", action="store_true", help="Resume training từ checkpoint_latest.pth")
    args = parser.parse_args()
    train(dataset_name=args.dataset, lr=args.lr, epochs=args.epochs, scheduler_name=args.scheduler,
          eta_min=args.eta_min, resume=args.resume, test_dataset_name=args.test_dataset)
