"""
Inference Module: Open-set classification với unified CNN-GRU model.

Pipeline (theo Train_model.md Bước 6 — Open-set inference):
    1. Load unified model train trên processed_data/merged/ (4 lớp known)
    2. Forward → softmax → ŷ ∈ R^4
    3. p_max = max(ŷ); k* = argmax(ŷ)
    4. Nếu p_max ≥ τ → predict k* ∈ {Normal, IR, OR, Ball}
       Nếu p_max < τ → predict class 4 = None (lỗi không xác định)

Output cuối: 5 lớp khả dĩ (0..4), nhưng softmax head chỉ có 4 neuron;
class None sinh ra bằng threshold rule, không qua training.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from train_model import CNNGRU
from process_data import sliding_window, zscore_normalize

SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))


# LABEL SPACE
# 4 lớp known dùng cho training (label space của merged model)
KNOWN_LABELS = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Outer Race Fault",
    3: "Ball Fault",
}

# Class 4 = None (open-set), không tham gia training
NONE_LABEL_ID = 4
NONE_LABEL = "None (lỗi không xác định)"

# Output cuối có 5 lớp khả dĩ
ALL_LABELS = {**KNOWN_LABELS, NONE_LABEL_ID: NONE_LABEL}

NUM_KNOWN_CLASSES = 4
DEFAULT_TAU = 0.7   # Threshold cho open-set, tune trên val nếu cần


# OPEN-SET CLASSIFIER
class OpenSetClassifier:
    """
    Phân loại open-set với 1 model unified.

    Train: 4 lớp known (Normal/IR/OR/Ball) trên processed_data/merged/
    Infer: 5 outcome (4 known + 1 None) qua threshold rule.
    """

    def __init__(self, model_path=None, tau=DEFAULT_TAU, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tau = tau

        if model_path is None:
            model_path = os.path.join(
                SOFTWARE_DIR, "saved_models", "merged", "cnn_gru_merged.pth"
            )

        self.model = self._load_model(model_path)
        print(f"OpenSetClassifier loaded on {self.device}")
        print(f"  Model: {model_path}")
        print(f"  Threshold τ = {tau}")

    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        num_classes = checkpoint['num_classes']
        if num_classes != NUM_KNOWN_CLASSES:
            print(f"  [Warning] Model có {num_classes} lớp, kỳ vọng {NUM_KNOWN_CLASSES}")
        model = CNNGRU(num_classes=num_classes).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict(self, signal, return_detail=False):
        """
        Phán đoán lỗi vòng bi từ tín hiệu rung (đã z-score normalize).

        Args:
            signal: numpy array (2048,) hoặc (B, 2048)
            return_detail: True để trả thêm probabilities + p_max

        Returns:
            dict (1 sample) hoặc list[dict] (batch) với:
                - label: tên lỗi
                - label_id: index 0..4 (4 = None)
                - confidence: p_max
                - probabilities: dict 4 lớp known
                - is_none: True nếu p_max < τ
        """
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32)
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        signal = signal.to(self.device)

        with torch.no_grad():
            logits = self.model(signal)              # (B, 4)
            probs = F.softmax(logits, dim=1)         # (B, 4)
            p_max, k_star = probs.max(dim=1)         # (B,), (B,)

        probs_np = probs.cpu().numpy()
        p_max_np = p_max.cpu().numpy()
        k_star_np = k_star.cpu().numpy()

        results = []
        for i in range(signal.shape[0]):
            is_none = p_max_np[i] < self.tau
            if is_none:
                pred_id = NONE_LABEL_ID
                pred_label = NONE_LABEL
            else:
                pred_id = int(k_star_np[i])
                pred_label = KNOWN_LABELS[pred_id]

            result = {
                'label': pred_label,
                'label_id': pred_id,
                'confidence': float(p_max_np[i]),
                'is_none': bool(is_none),
                'probabilities': {
                    KNOWN_LABELS[k]: float(probs_np[i, k])
                    for k in range(NUM_KNOWN_CLASSES)
                },
            }
            if return_detail:
                result['logits'] = logits[i].cpu().numpy().tolist()
                result['threshold'] = self.tau

            results.append(result)

        return results[0] if len(results) == 1 else results

    def predict_from_raw(self, raw_signal, window_size=2048, step_size=1024):
        """
        Phán đoán từ tín hiệu thô (chưa cắt window).

        Cắt thành windows → z-score → predict từng window → voting đa số.
        """
        windows = sliding_window(raw_signal, window_size, step_size)
        windows = zscore_normalize(windows)

        results = self.predict(windows)
        if isinstance(results, dict):
            results = [results]

        # Voting + trung bình probability
        votes = {}
        total_probs = np.zeros(NUM_KNOWN_CLASSES)
        none_count = 0

        for r in results:
            label = r['label']
            votes[label] = votes.get(label, 0) + 1
            for k_name, p in r['probabilities'].items():
                idx = [i for i, name in KNOWN_LABELS.items() if name == k_name][0]
                total_probs[idx] += p
            if r['is_none']:
                none_count += 1

        final_label = max(votes, key=votes.get)
        avg_probs = total_probs / len(results)
        avg_p_max = avg_probs.max()

        return {
            'label': final_label,
            'confidence': float(avg_p_max),
            'num_windows': len(results),
            'none_ratio': none_count / len(results),
            'votes': votes,
            'avg_probabilities': {
                KNOWN_LABELS[k]: float(avg_probs[k])
                for k in range(NUM_KNOWN_CLASSES)
            },
        }


# DEMO
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Open-set inference với unified CNN-GRU")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Threshold cho class None (default: 0.7)")
    parser.add_argument("--dataset", type=str, default="merged",
                        choices=["cwru", "mfpt", "paderborn", "merged"],
                        help="Dataset test (default: merged)")
    args = parser.parse_args()

    print("=" * 65)
    print("  OPEN-SET INFERENCE - UNIFIED CNN-GRU MODEL")
    print("=" * 65)

    classifier = OpenSetClassifier(tau=args.tau)

    test_path = os.path.join(SOFTWARE_DIR, "processed_data", args.dataset, "X_test.npy")
    label_path = os.path.join(SOFTWARE_DIR, "processed_data", args.dataset, "Y_test.npy")

    if not os.path.exists(test_path):
        print(f"\nKhông tìm thấy {test_path}")
        print("Chạy process_data.py --dataset all trước, rồi train_model.py --dataset merged.")
        exit(1)

    X_test = np.load(test_path)
    Y_test = np.load(label_path)

    print(f"\n{'─' * 65}")
    print(f"  TEST VỚI {args.dataset.upper()} ({len(Y_test)} samples)")
    print(f"{'─' * 65}")

    results = classifier.predict(X_test)
    if isinstance(results, dict):
        results = [results]

    correct = 0
    none_count = 0
    confusion = {}  # (true, pred) → count

    for i, r in enumerate(results):
        true_id = int(Y_test[i])
        pred_id = r['label_id']

        if r['is_none']:
            none_count += 1
        if pred_id == true_id:
            correct += 1

        key = (true_id, pred_id)
        confusion[key] = confusion.get(key, 0) + 1

    acc = correct / len(Y_test)
    none_ratio = none_count / len(Y_test)

    print(f"  Accuracy (so với nhãn thật): {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Tỉ lệ predict None (open-set): {none_ratio:.4f} ({none_ratio * 100:.2f}%)")
    print(f"  Threshold τ = {args.tau}")

    print(f"\n  Ví dụ dự đoán (5 mẫu đầu):")
    for i in range(min(5, len(results))):
        r = results[i]
        true_name = KNOWN_LABELS.get(int(Y_test[i]), f"Class {Y_test[i]}")
        marker = "✗" if r['label_id'] != int(Y_test[i]) else "✓"
        print(f"    {marker} Sample {i}: True={true_name:25s} → Pred={r['label']:25s} (p={r['confidence']:.4f})")

    print(f"\n{'=' * 65}")
    print("  DEMO HOÀN TẤT")
    print(f"{'=' * 65}")
