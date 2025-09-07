# total_pipeline.py
import argparse
from pathlib import Path
import subprocess
import sys
import os
import pandas as pd
import yaml

# ---- 기본 드롭 컬럼(이미 주신 목록) ----
DROP_COLS_DEFAULT = [
    "t_target_ns", "odom_src_ns", "odom_v_mps", "odom_w_rps",
    "imu_src_ns", "imu_gyro_z_rps", "imu_accel_x_mps2",
]

REQUIRED = ["file", "stamp_ns", "twist_linear_x_mps", "twist_angular_z_rps"]

def prepare_one(src_csv: Path, tag: str, drop_cols):
    """
    1) src_csv 로드 -> 불필요 컬럼 제거
    2) 필수 컬럼만 남기고 NaN 드롭, stamp_ns로 정렬
    3) 같은 폴더에 '<stem>__<tag>__prepared.csv' 저장
    4) 준비된 CSV 절대경로 반환
    """
    if not src_csv.exists():
        raise FileNotFoundError(f"CSV not found: {src_csv}")

    df = pd.read_csv(src_csv)

    # 드롭(없는 건 무시)
    drop_cols = [c for c in (drop_cols or []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 필수 컬럼 체크
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{src_csv}: required columns missing: {missing}")

    df = df.dropna(subset=REQUIRED).sort_values("stamp_ns").reset_index(drop=True)

    out_csv = src_csv.with_name(f"{src_csv.stem}__{tag}__prepared.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote: {out_csv}  (rows={len(df)}, cols={list(df.columns)})")
    return out_csv.resolve()

def write_yaml(csv_paths, out_yaml: Path):
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump({"runs": [str(p) for p in csv_paths]}, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] wrote YAML: {out_yaml}")

def smoke_test(train_yaml: Path, rgb_lane_script_dir: Path, seq_len: int, img_size: int, smoke_n: int):
    """
    간단 스모크: 각 CSV 앞쪽 smoke_n개만 사용하도록 임시 mini CSV/YAML 만들고
    Dataset/Model을 import해 한 배치 forward.
    """
    # rgb_lane_train_eval.py 모듈 임포트 경로 추가
    sys.path.insert(0, str(rgb_lane_script_dir))
    from rgb_lane_train_eval import LaneControlSeqDataset, RGB2CtrlNoStatus, make_bin_centers
    import torch

    # mini 구성
    with open(train_yaml, "r") as f:
        train_runs = yaml.safe_load(f)["runs"]

    mini_dir = train_yaml.parent / "mini"
    mini_dir.mkdir(parents=True, exist_ok=True)
    mini_csvs = []

    for csv_path in train_runs:
        src = Path(csv_path)
        df = pd.read_csv(src)
        df = df.head(smoke_n).reset_index(drop=True)  # 앞쪽 N개
        mini_csv = mini_dir / f"{src.stem}__mini.csv"
        df.to_csv(mini_csv, index=False)
        mini_csvs.append(mini_csv)

    mini_yaml = mini_dir / "train_list.yaml"
    write_yaml(mini_csvs, mini_yaml)

    # 로더 & 모델 한 번
    vmin, vmax, Kv = 0.0, 8.0, 21
    wmin, wmax, Kw = -0.6, 0.6, 41
    ds = LaneControlSeqDataset(
        str(mini_yaml),
        seq_len=seq_len,
        stride=1,
        img_size=img_size,
        train=True,
        use_class_label=True,
        v_bin_cfg=(vmin, vmax, Kv),
        w_bin_cfg=(wmin, wmax, Kw),
        v_soft_sigma=0.4, w_soft_sigma=0.03,
        drop_missing_images=False,
        img_root="",  # 경로는 CSV 위치 기준
    )
    if len(ds) == 0:
        raise RuntimeError("Smoke dataset has zero samples. Increase --smoke-n or check CSV/file paths.")
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    x = batch["images"]  # [B,T,3,H,W]
    print(f"[SMOKE] batch images shape: {tuple(x.shape)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGB2CtrlNoStatus(K_v=Kv, K_w=Kw, hidden=256).to(device).eval()
    with torch.inference_mode():
        logits_v, logits_w = model(x.to(device))
    print(f"[SMOKE] model outputs: v {tuple(logits_v.shape)}, w {tuple(logits_w.shape)}")
    print("✅ Smoke test passed.")

def run_training(train_yaml: Path, val_yaml: Path, train_script: Path, args):
    """
    rgb_lane_train_eval.py를 서브프로세스로 호출하여 학습 실행
    """
    cmd = [
        sys.executable, str(train_script),
        "--train_yaml", str(train_yaml),
        "--val_yaml",   str(val_yaml),
        "--epochs",     str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--seq_len",    str(args.seq_len),
        "--stride",     str(args.stride),
        "--img_size",   str(args.img_size),
        "--vmin",       str(args.vmin),
        "--vmax",       str(args.vmax),
        "--Kv",         str(args.Kv),
        "--wmin",       str(args.wmin),
        "--wmax",       str(args.wmax),
        "--Kw",         str(args.Kw),
        "--out_dir",    str(args.out_dir),
    ]
    if args.amp:
        cmd.append("--amp")
    if args.resume:
        cmd += ["--resume", str(args.resume)]

    print("[TRAIN] launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Total pipeline: prepare CSVs -> YAML -> (optional) smoke -> (optional) train")
    ap.add_argument("--dark-fast",   required=True, help="dark & fast labels.csv")
    ap.add_argument("--bright-fast", required=True, help="bright & fast labels.csv")
    ap.add_argument("--dark-slow",   required=True, help="dark & slow labels.csv")
    ap.add_argument("--bright-slow", required=True, help="bright & slow labels.csv")

    ap.add_argument("--drop-cols", nargs="*", default=DROP_COLS_DEFAULT, help="columns to drop")
    ap.add_argument("--lists-dir", required=True, help="where to write train_list.yaml and val_list.yaml")

    # smoke
    ap.add_argument("--smoke", action="store_true", help="run a quick smoke test after preparing")
    ap.add_argument("--smoke-n", type=int, default=50, help="how many rows from each CSV for smoke test")
    ap.add_argument("--seq-len", type=int, default=8, help="sequence length (also used for training if --train)")
    ap.add_argument("--img-size", type=int, default=224)

    # train
    ap.add_argument("--train", action="store_true", help="launch training via rgb_lane_train_eval.py")
    ap.add_argument("--train-script", type=str, default="", help="path to rgb_lane_train_eval.py")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--Kv",   type=int,   default=21)
    ap.add_argument("--wmin", type=float, default=-0.6)
    ap.add_argument("--wmax", type=float, default= 0.6)
    ap.add_argument("--Kw",   type=int,   default=41)
    ap.add_argument("--out-dir", type=str, default="./runs/lane_cls")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    # 1) 각 CSV 정리 & 같은 폴더에 prepared 저장
    pairs = [
        (Path(args.dark_fast),   "dark_fast"),
        (Path(args.bright_fast), "bright_fast"),
        (Path(args.dark_slow),   "dark_slow"),
        (Path(args.bright_slow), "bright_slow"),
    ]
    prepared = []
    for src, tag in pairs:
        prepared.append(prepare_one(src, tag, args.drop_cols))

    # 2) train/val YAML 생성
    lists_dir = Path(args.lists_dir).resolve()
    train_yaml = lists_dir / "train_list.yaml"
    val_yaml   = lists_dir / "val_list.yaml"
    train_runs = prepared[:2]  # fast 2개
    val_runs   = prepared[2:]  # slow 2개
    write_yaml(train_runs, train_yaml)
    write_yaml(val_runs,   val_yaml)

    # 3) (옵션) 스모크 테스트
    if args.smoke:
        # rgb_lane_train_eval.py가 있는 디렉토리를 모듈 경로에 추가해야 import 가능
        if not args.train_script:
            print("[SMOKE] --train-script 경로를 지정하면 동일 디렉토리에서 Dataset/Model을 import해 스모크를 수행합니다.")
            sys.exit(1)
        rgb_dir = Path(args.train_script).resolve().parent
        smoke_test(train_yaml, rgb_dir, seq_len=max(2, min(args.seq_len, 8)), img_size=args.img_size, smoke_n=args.smoke_n)

    # 4) (옵션) 학습 실행
    if args.train:
        if not args.train_script:
            print("[TRAIN] --train-script 경로가 필요합니다 (rgb_lane_train_eval.py).")
            sys.exit(1)
        run_training(train_yaml, val_yaml, Path(args.train_script).resolve(), args)

    print("\nDone.")

if __name__ == "__main__":
    main()
