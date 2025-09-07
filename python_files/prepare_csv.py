# prepare_4_csvs_in_place.py
from pathlib import Path
import pandas as pd

# ▼ 여기에 너의 4개 원본 CSV 경로를 그대로 넣어두면 됩니다.
#   (지금 예시는 네가 올렸던 경로 그대로 사용)
CSV_PATHS = [
    # light: dark
    ("/home/takeout/e2e_limo_ws/src/E2E_LIMO/data_collect/light_low/speed_high/e2e_dataset_live/labels.csv",  "dark_fast"),
    ("/home/takeout/e2e_limo_ws/src/E2E_LIMO/data_collect/light_low/speed_high_backward/e2e_dataset_live/labels.csv","dark_fast_backward"),

    # light: bright
    ("/home/takeout/e2e_limo_ws/src/E2E_LIMO/data_collect/light_normal/speed_high/e2e_dataset_live/labels.csv",  "bright_fast"),
    ("/home/takeout/e2e_limo_ws/src/E2E_LIMO/data_collect/light_normal/speed_high_backward/e2e_dataset_live/labels.csv","bright_fast_backward"),
    
    # light: crazy
    ("/home/takeout/e2e_limo_ws/src/E2E_LIMO/data_collect/light_crazy/speed_high/e2e_dataset_live/labels.csv","crazy_fast"),
    ("/home/takeout/e2e_limo_ws/src/E2E_LIMO/data_collect/light_crazy/speed_high_backward/e2e_dataset_live/labels.csv","crazy_fast_backward"),
]

# 필요 없는 컬럼 (네가 준 목록)
DROP_COLS = [
    "t_target_ns", "odom_src_ns", "odom_v_mps", "odom_w_rps",
    "imu_src_ns", "imu_gyro_z_rps", "imu_accel_x_mps2",
]

# 최소로 남겨야 하는 컬럼(필수)
REQUIRED = ["file", "stamp_ns", "twist_linear_x_mps", "twist_angular_z_rps"]

def prepare_one(src_csv: str, tag: str) -> Path:
    src = Path(src_csv).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"CSV not found: {src}")

    df = pd.read_csv(src)

    # 불필요 컬럼 제거(없는 건 무시)
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 필수 컬럼 확인 + 정렬/NaN 제거
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{src}: 필수 컬럼 누락: {missing}")
    df = df.dropna(subset=REQUIRED).sort_values("stamp_ns").reset_index(drop=True)

    # 같은 위치에 저장 (파일명: <원본stem>__<tag>__prepared.csv)
    out_csv = src.with_name(f"{src.stem}__{tag}__prepared.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote: {out_csv}  (rows={len(df)}, cols={list(df.columns)})")
    return out_csv

def main():
    out_paths = []
    for path, tag in CSV_PATHS:
        out_paths.append(prepare_one(path, tag))

    print("\n==== Prepared CSVs ====")
    for p in out_paths:
        print(" -", p)

    # 참고: 이후 YAML에는 위 경로들을 그대로 넣으면 됩니다.
    # runs:
    #   - <dark_fast prepared csv>
    #   - <bright_fast prepared csv>
    # (val)
    #   - <dark_slow prepared csv>
    #   - <bright_slow prepared csv>

if __name__ == "__main__":
    main()
