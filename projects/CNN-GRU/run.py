"""
Runner: huấn luyện CNN-GRU theo kịch bản.

Cách dùng:
    python run.py                                     # all 7 jobs, 100 epoch
    python run.py --jobs individual                   # cwru / mfpt / paderborn
    python run.py --jobs merged                       # merged dataset
    python run.py --jobs cross                        # 3 cross-dataset jobs
    python run.py --jobs all --epochs 200 --resume    # toàn bộ, tiếp tục từ checkpoint
    python run.py --jobs cross --epochs 200 --resume  # cross, tiếp tục
"""

import argparse
import time
from train_model import train, LEARNING_RATE, NUM_EPOCHS

INDIVIDUAL_JOBS = [
    ("cwru",      None,          "CWRU individual"),
    ("mfpt",      None,          "MFPT individual"),
    ("paderborn", None,          "Paderborn individual"),
]

MERGED_JOBS = [
    ("merged",    None,          "Merged individual"),
]

CROSS_JOBS = [
    ("cwru",      "mfpt",        "Cross: CWRU → MFPT"),
    ("mfpt",      "paderborn",   "Cross: MFPT → Paderborn"),
    ("paderborn", "cwru",        "Cross: Paderborn → CWRU"),
]

PRESETS = {
    "individual": INDIVIDUAL_JOBS,
    "merged":     MERGED_JOBS,
    "cross":      CROSS_JOBS,
    "all":        INDIVIDUAL_JOBS + MERGED_JOBS + CROSS_JOBS,
}


def run_jobs(jobs, epochs, resume, lr, scheduler, eta_min):
    results = []
    total = len(jobs)

    for idx, (train_ds, test_ds, desc) in enumerate(jobs, 1):
        print(f"\n{'#' * 65}", flush=True)
        print(f"  JOB {idx}/{total}: {desc}", flush=True)
        print(f"{'#' * 65}", flush=True)
        t0 = time.time()
        try:
            _, acc = train(
                dataset_name=train_ds,
                test_dataset_name=test_ds,
                epochs=epochs,
                resume=resume,
                lr=lr,
                scheduler_name=scheduler,
                eta_min=eta_min,
            )
            elapsed = time.time() - t0
            results.append((desc, acc, elapsed, None))
            print(f"\n  [OK] {desc} — Acc={acc:.4f} ({elapsed/60:.1f} min)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            elapsed = time.time() - t0
            results.append((desc, None, elapsed, str(e)))
            print(f"\n  [FAIL] {desc} — {e}", flush=True)

    print(f"\n\n{'=' * 65}", flush=True)
    print("  TỔNG KẾT", flush=True)
    print(f"{'=' * 65}", flush=True)
    print(f"  {'Kịch bản':<35} {'Test Acc':>10} {'Thời gian':>12}  Trạng thái", flush=True)
    print(f"  {'-' * 65}", flush=True)
    for desc, acc, elapsed, err in results:
        status = "OK" if err is None else f"FAIL: {err[:30]}"
        acc_str = f"{acc:.4f}" if acc is not None else "—"
        print(f"  {desc:<35} {acc_str:>10} {elapsed/60:>10.1f}m  {status}", flush=True)
    print(f"{'=' * 65}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runner: huấn luyện CNN-GRU theo kịch bản",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python run.py                                      # all 7 jobs, 100 epoch
  python run.py --jobs individual --epochs 100       # 3 individual jobs
  python run.py --jobs cross --epochs 200 --resume   # cross-dataset, tiếp tục
  python run.py --jobs all --epochs 200 --resume     # toàn bộ 7 jobs, tiếp tục
        """,
    )
    parser.add_argument(
        "--jobs",
        choices=list(PRESETS),
        default="all",
        help="Nhóm kịch bản: individual | merged | cross | all (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Số epoch tối đa (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục từ checkpoint_latest.pth nếu có",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate khởi đầu (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "onecycle", "cosine"],
        default="plateau",
        help="LR scheduler (default: plateau)",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=1e-6,
        help="Min LR cho cosine scheduler (default: 1e-6)",
    )
    args = parser.parse_args()

    run_jobs(
        jobs=PRESETS[args.jobs],
        epochs=args.epochs,
        resume=args.resume,
        lr=args.lr,
        scheduler=args.scheduler,
        eta_min=args.eta_min,
    )
