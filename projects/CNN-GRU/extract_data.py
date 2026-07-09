"""
Feature Extraction Module cho bearing fault diagnosis.

Trích xuất đặc trưng từ tín hiệu rung đã cắt window (2048 điểm) thành
SEQUENCE đặc trưng để đưa vào GRU (T > 1, GRU thực sự có việc làm).

Kiến trúc 2 nhánh song song (theo extract_data.md):
    Nhánh A (1D temporal): Conv1D pipeline → AdaptiveAvgPool1d(T=32)
                          → transpose → (B, 32, 128)
    Nhánh B (2D spectral): STFT → Spectrogram → Conv2D pipeline
                          → mean pool freq → AdaptiveAvgPool1d(T=32)
                          → transpose → (B, 32, 128)
    Output: concat theo channel → (B, 32, 256)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# CẤU HÌNH / HYPERPARAMETERS
WINDOW_SIZE = 2048

# Nhánh A (1D temporal)
BRANCH_A_KERNEL = 7          # kernel_size cho Conv1D
BRANCH_A_STRIDE_DOWN = 2     # stride cho Strided Convolution (giảm chiều)

# Nhánh B (2D spectral - STFT)
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
# → Spectrogram shape: (n_fft//2 + 1, time_frames) = (513, ~9)

# Output
FEATURE_DIM = 128            # kích thước channel của f_A và f_B
SEQUENCE_LEN = 32            # T cố định cho cả 2 nhánh (AdaptiveAvgPool1d)


# NHÁNH A: VECTOR 1D - TRỤC THỜI GIAN
class BranchA(nn.Module):
    """
    Pipeline trích xuất đặc trưng 1D (temporal features).

    Bước 1: Normalize             → InstanceNorm1d (chuẩn hóa per-sample)
    Bước 2: Conv1D lọc            → học filter tự động, bắt pattern ngắn hạn
    Bước 3: GELU + BN             → phi tuyến + ổn định dữ liệu
    Bước 4: Strided Conv          → giảm kích thước theo thời gian
    Bước 5: Stack Conv1D          → 3 block Conv+BN+GELU để học pattern sâu hơn
    Bước 6: AdaptiveAvgPool1d(T=32) + transpose → sequence (B, 32, 128) cho GRU

    Input:  (B, 1, 2048) hoặc (B, 2048)
    Output: (B, 32, 128) = sequence f_A
    """

    def __init__(self):
        super(BranchA, self).__init__()

        k = BRANCH_A_KERNEL
        p = k // 2  # padding để giữ kích thước (same padding)
        s = BRANCH_A_STRIDE_DOWN

        # Bước 1: Normalize per-sample
        # InstanceNorm1d chuẩn hóa từng sample riêng biệt
        # giúp model ít phụ thuộc vào biên độ tuyệt đối
        self.norm_input = nn.InstanceNorm1d(1)

        # Bước 2 + 3: Conv1D lọc tín hiệu + GELU + BatchNorm
        # Conv1d: y(t) = Σ x(t+k) · w(k) → học filter tự động, bắt pattern ngắn hạn
        self.conv1 = nn.Conv1d(1, 32, kernel_size=k, padding=p)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(32)

        # Bước 4: Strided Convolution → giảm chiều thời gian
        # Output size = ⌊(N - K) / S⌋ + 1
        # Tăng vùng dữ liệu mà model nhìn thấy (receptive field)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=k, stride=s, padding=p)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(64)

        # Bước 5: Stack Conv1D - xếp chồng nhiều lớp để học đặc trưng từ đơn giản đến phức tạp
        # Block 1: Conv + BN + GELU → tổng hợp đặc trưng cơ bản từ bước 2
        self.conv3 = nn.Conv1d(64, 128, kernel_size=k, padding=p)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(128)

        # Block 2: Conv + BN + GELU → học pattern phức tạp hơn, nắm mối quan hệ giữa các lớp
        self.conv4 = nn.Conv1d(128, 128, kernel_size=k, padding=p)
        self.gelu4 = nn.GELU()
        self.bn4 = nn.BatchNorm1d(128)

        # Block 3: Conv + BN + GELU → pattern sâu nhất, receptive field rộng nhất
        self.conv5 = nn.Conv1d(128, 128, kernel_size=k, padding=p)
        self.gelu5 = nn.GELU()
        self.bn5 = nn.BatchNorm1d(128)

        # Bước 6: AdaptiveAvgPool1d(T=32) — KHÔNG dùng GAP
        # Giữ chiều thời gian T=32 để GRU có sequence thật sự để học
        self.adaptive_pool = nn.AdaptiveAvgPool1d(SEQUENCE_LEN)

    def forward(self, x):
        """
        Args:
            x: (B, 2048) hoặc (B, 1, 2048)
        Returns:
            f_A: (B, T=32, feature_dim=128) — sequence cho GRU
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 2048) → (B, 1, 2048)

        # Bước 1: Normalize
        x = self.norm_input(x)

        # Bước 2 + 3: Conv1D + GELU + BatchNorm
        x = self.bn1(self.gelu1(self.conv1(x)))      # (B, 32, 2048)

        # Bước 4: Strided Conv (giảm chiều thời gian)
        x = self.bn2(self.gelu2(self.conv2(x)))      # (B, 64, 1024)

        # Bước 5: Stack Conv1D — 3 block Conv+BN+GELU
        x = self.gelu3(self.bn3(self.conv3(x)))      # (B, 128, 1024)
        x = self.gelu4(self.bn4(self.conv4(x)))      # (B, 128, 1024)
        x = self.gelu5(self.bn5(self.conv5(x)))      # (B, 128, 1024)

        # Bước 6: AdaptiveAvgPool1d(32) + transpose → sequence
        x = self.adaptive_pool(x)                    # (B, 128, 32)
        f_a = x.transpose(1, 2)                      # (B, 32, 128)

        return f_a


# NHÁNH B: VECTOR 2D - TRỤC TẦN SỐ
class BranchB(nn.Module):
    """
    Pipeline trích xuất đặc trưng 2D (spectral features).

    Bước 1: STFT                  → chia tín hiệu thành frame, biến đổi Fourier
    Bước 2: Power Spec + log      → nén dynamic range, làm nổi dải yếu
    Bước 3: Normalize             → Sample normalize + Frequency normalize
    Bước 4: Conv2D + MaxPool      → học pattern 2D theo (thời gian, tần số)
    Bước 5: Mean pool tần số + AdaptiveAvgPool1d(T=32) + transpose
            → sequence (B, 32, 128) cho GRU

    Input:  (B, 1, 2048) hoặc (B, 2048)
    Output: (B, 32, 128) = sequence f_B
    """

    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
        super(BranchB, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Tạo window function cho STFT (Hann window)
        self.register_buffer('window', torch.hann_window(win_length))

        # Bước 4: Conv2D blocks → học pattern 2D theo (thời gian, tần số)
        self.conv_blocks = nn.Sequential(
            # Block 1: Conv2D + BN + GELU → pattern cơ bản: cạnh, vệt sáng, dải tần nổi bật
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # MaxPool2D(2x2) → giảm tần số + thời gian còn nửa; giữ đặc trưng nổi bật nhất
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: Conv2D + BN + GELU → pattern phức tạp hơn, kết hợp đặc trưng lớp trước
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # MaxPool(2x1) → chỉ giảm tần số, giữ nguyên thời gian
            # (tránh thời gian quá nhỏ → quá ít thông tin cho GAP)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 3: Conv2D + BN + GELU → pattern cao nhất, mỗi điểm đại diện vùng rộng của spectrogram gốc
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Bước 5: Mean pool tần số → AdaptiveAvgPool1d trên trục thời gian
        # KHÔNG dùng GAP toàn bộ — giữ chiều thời gian T=32 cho GRU
        self.time_pool = nn.AdaptiveAvgPool1d(SEQUENCE_LEN)

    def _compute_spectrogram(self, x):
        """
        Bước 1: STFT
            X(τ, f) = ∫ x(t) · w(t-τ) · e^(-j2πft) dt
            Chia tín hiệu thành các frame ngắn, mỗi frame biến đổi Fourier
            để phân tích tín hiệu theo thời gian và tần số.

        Bước 2: Power Spectrogram + log(dB)
            S = log(1 + |STFT|²)
            Nén dynamic range + làm nổi dải yếu

        Args:
            x: (B, 2048) - tín hiệu 1D
        Returns:
            spec: (B, 1, F, T) - spectrogram 2D
                  F = n_fft//2 + 1 = 129 (trục tần số)
                  T = time frames (trục thời gian)
        """
        # STFT: output shape (B, n_fft//2+1, T, 2) cho real/imag
        stft_out = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )  # (B, F, T) complex

        # Power Spectrogram: |STFT|²
        power = stft_out.abs() ** 2  # (B, F, T)

        # Log scale (dB): log(1 + |STFT|²) → nén dynamic range
        spec = torch.log1p(power)  # (B, F, T)

        # Thêm channel dimension cho Conv2D
        spec = spec.unsqueeze(1)  # (B, 1, F, T)

        return spec

    def _normalize_spectrogram(self, spec):
        """
        Bước 3: Ổn định dữ liệu

        Sample normalize: S' = (S - μ) / σ
            Chuẩn hóa toàn bộ spectrogram của từng sample

        Frequency normalize: chuẩn hóa dải tần riêng biệt
            Giúp CNN không bỏ qua dữ liệu vùng tối (năng lượng thấp)

        Args:
            spec: (B, 1, F, T)
        Returns:
            spec_norm: (B, 1, F, T) - đã chuẩn hóa
        """
        # Sample normalize
        # Tính mean/std trên toàn bộ spectrogram của mỗi sample
        B, C, F, T = spec.shape
        spec_flat = spec.reshape(B, -1)  # (B, C*F*T)
        mean_s = spec_flat.mean(dim=1, keepdim=True)  # (B, 1)
        std_s = spec_flat.std(dim=1, keepdim=True) + 1e-8  # (B, 1)
        spec_flat = (spec_flat - mean_s) / std_s
        spec = spec_flat.reshape(B, C, F, T)  # (B, 1, F, T)

        # Frequency normalize
        # Chuẩn hóa theo từng dải tần (mỗi hàng trong spectrogram)
        # S(f, t) → chuẩn hóa riêng biệt cho mỗi tần số f
        mean_f = spec.mean(dim=-1, keepdim=True)  # (B, 1, F, 1)
        std_f = spec.std(dim=-1, keepdim=True) + 1e-8  # (B, 1, F, 1)
        spec = (spec - mean_f) / std_f

        return spec

    def forward(self, x):
        """
        Args:
            x: (B, 2048) hoặc (B, 1, 2048)
        Returns:
            f_B: (B, T=32, feature_dim=128) — sequence cho GRU
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, 2048) → (B, 2048)

        # Bước 1 + 2: STFT → Power Spectrogram + log
        spec = self._compute_spectrogram(x)        # (B, 1, F, T)

        # Bước 3: Normalize
        spec = self._normalize_spectrogram(spec)   # (B, 1, F, T)

        # Bước 4: Conv2D blocks + MaxPool
        x = self.conv_blocks(spec)                 # (B, 128, F', T')

        # Bước 5a: Mean pool theo trục tần số (dim=2)
        x = x.mean(dim=2)                          # (B, 128, T')

        # Bước 5b: AdaptiveAvgPool1d → cố định T=32 khớp với nhánh A
        x = self.time_pool(x)                      # (B, 128, 32)

        # Bước 5c: Transpose để có sequence format
        f_b = x.transpose(1, 2)                    # (B, 32, 128)

        return f_b


# FEATURE EXTRACTOR: TỔNG HỢP 2 NHÁNH
class FeatureExtractor(nn.Module):
    """
    Module trích xuất đặc trưng song song 2 nhánh, output dạng sequence cho GRU.

    Nhánh A (temporal): tín hiệu 1D → Conv1D pipeline → (B, 32, 128)
    Nhánh B (spectral): STFT → Conv2D pipeline → (B, 32, 128)

    Concat theo channel: (B, 32, 256) — sẵn sàng cho GRU

    Input:  (B, 2048) hoặc (B, 1, 2048)
    Output: (B, 32, 256)
    """

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.branch_a = BranchA()
        self.branch_b = BranchB()
        self.feature_dim = FEATURE_DIM
        self.sequence_len = SEQUENCE_LEN

    def forward(self, x):
        """
        Args:
            x: (B, 2048) hoặc (B, 1, 2048)
        Returns:
            features: (B, T=32, C_A + C_B = 256) — sequence cho GRU
        """
        f_a = self.branch_a(x)                    # (B, 32, 128)
        f_b = self.branch_b(x)                    # (B, 32, 128)

        # Concat theo channel (dim=2)
        features = torch.cat([f_a, f_b], dim=2)   # (B, 32, 256)

        return features

    def extract_separate(self, x):
        """Trích xuất f_A và f_B riêng biệt (debug/visualization)."""
        f_a = self.branch_a(x)
        f_b = self.branch_b(x)
        return f_a, f_b


# DEMO / VERIFICATION
if __name__ == "__main__":
    SOFTWARE_DIR = os.path.dirname(os.path.abspath(__file__))

    print("=" * 65)
    print("  FEATURE EXTRACTION MODULE - VERIFICATION")
    print("=" * 65)

    # Khởi tạo model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().to(device)

    # In kiến trúc
    print(f"\nDevice: {device}")
    print(f"Feature dim mỗi nhánh: {FEATURE_DIM}")
    print(f"Sequence length T: {SEQUENCE_LEN}")
    print(f"Output shape: (B, {SEQUENCE_LEN}, {FEATURE_DIM * 2})")

    # Đếm parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load data thực từ CWRU
    cwru_path = os.path.join(SOFTWARE_DIR, "processed_data", "cwru", "X_train.npy")

    if os.path.exists(cwru_path):
        print(f"\n{'─' * 65}")
        print("  TEST VỚI DATA THỰC (CWRU)")
        print(f"{'─' * 65}")

        X_train = np.load(cwru_path)
        batch_size = 32
        batch = X_train[:batch_size]

        print(f"Loaded: {cwru_path}")
        print(f"X_train shape: {X_train.shape}")
        print(f"Batch shape: {batch.shape}")

        # Chuyển sang tensor
        x = torch.tensor(batch, dtype=torch.float32).to(device)
        print(f"Input tensor: {x.shape}")

        # Chạy qua model (eval mode)
        model.eval()
        with torch.no_grad():
            f_a, f_b = model.extract_separate(x)
            features = model(x)

        print(f"\n  Nhánh A (temporal 1D):")
        print(f"    f_A shape: {f_a.shape}")
        print(f"    f_A range: [{f_a.min().item():.4f}, {f_a.max().item():.4f}]")
        print(f"    f_A mean:  {f_a.mean().item():.4f}")
        print(f"    f_A has NaN: {torch.isnan(f_a).any().item()}")

        print(f"\n  Nhánh B (spectral 2D):")
        print(f"    f_B shape: {f_b.shape}")
        print(f"    f_B range: [{f_b.min().item():.4f}, {f_b.max().item():.4f}]")
        print(f"    f_B mean:  {f_b.mean().item():.4f}")
        print(f"    f_B has NaN: {torch.isnan(f_b).any().item()}")

        print(f"\n  Output (concat):")
        print(f"    features shape: {features.shape}")
        print(f"    features range: [{features.min().item():.4f}, {features.max().item():.4f}]")
        print(f"    features has NaN: {torch.isnan(features).any().item()}")
        print(f"    features has Inf: {torch.isinf(features).any().item()}")

        # Verify STFT spectrogram shape
        print(f"\n  STFT Config:")
        print(f"    n_fft={N_FFT}, hop_length={HOP_LENGTH}, win_length={WIN_LENGTH}")
        x_1d = x[:1].squeeze(0) if x.dim() == 2 else x[:1].squeeze(0).squeeze(0)
        with torch.no_grad():
            stft_out = torch.stft(
                x[:1] if x.dim() == 2 else x[:1].squeeze(1),
                n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                window=torch.hann_window(WIN_LENGTH).to(device),
                return_complex=True
            )
        print(f"    Spectrogram shape: {stft_out.shape}")
        print(f"    (B, F={stft_out.shape[1]}, T={stft_out.shape[2]})")

    else:
        print(f"\nKhông tìm thấy {cwru_path}")
        print("Chạy process_data.py trước để tạo dữ liệu.")

    # Test với random data
    print(f"\n{'─' * 65}")
    print("  TEST VỚI RANDOM DATA")
    print(f"{'─' * 65}")

    x_rand = torch.randn(32, WINDOW_SIZE).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x_rand)

    print(f"Input:  {x_rand.shape}")
    print(f"Output: {out.shape}")
    print(f"Expected: (32, {SEQUENCE_LEN}, {FEATURE_DIM * 2})")
    assert out.shape == (32, SEQUENCE_LEN, FEATURE_DIM * 2), f"Shape mismatch! Got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN!"
    assert not torch.isinf(out).any(), "Output contains Inf!"
    print("PASSED - All shape and value checks OK!")

    print(f"\n{'=' * 65}")
    print("  VERIFICATION COMPLETE")
    print(f"{'=' * 65}")
