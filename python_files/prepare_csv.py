# prepare_csv.py
from pathlib import Path
import pandas as pd

# original CSV path
CSV_PATHS = [
    # light: dark
    ("~/e2e_limo_ws/src/E2E_LIMO/data_collect/light_low/speed_high/e2e_dataset_live/labels.csv",  "dark_fast"),
    ("~/e2e_limo_ws/src/E2E_LIMO/data_collect/light_low/speed_high_backward/e2e_dataset_live/labels.csv","dark_fast_backward"),

    # light: bright
    ("~/e2e_limo_ws/src/E2E_LIMO/data_collect/light_normal/speed_high/e2e_dataset_live/labels.csv",  "bright_fast"),
    ("~/e2e_limo_ws/src/E2E_LIMO/data_collect/light_normal/speed_high_backward/e2e_dataset_live/labels.csv","bright_fast_backward"),
    
    # light: crazy
    ("~/e2e_limo_ws/src/E2E_LIMO/data_collect/light_crazy/speed_high/e2e_dataset_live/labels.csv","crazy_fast"),
    ("~/e2e_limo_ws/src/E2E_LIMO/data_collect/light_crazy/speed_high_backward/e2e_dataset_live/labels.csv","crazy_fast_backward"),
]

# columns for drop
DROP_COLS = [
    "t_target_ns", "odom_src_ns", "odom_v_mps", "odom_w_rps",
    "imu_src_ns", "imu_gyro_z_rps", "imu_accel_x_mps2",
]

# required columns
REQUIRED = ["file", "stamp_ns", "twist_linear_x_mps", "twist_angular_z_rps"]

def prepare_one(src_csv: str, tag: str) -> Path:
    src = Path(src_csv).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"CSV not found: {src}")

    df = pd.read_csv(src)

    # remove columns
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # sort by timestamp(stamp_ns)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{src}: 필수 컬럼 누락: {missing}")
    df = df.dropna(subset=REQUIRED).sort_values("stamp_ns").reset_index(drop=True)

    # save new csv
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

    # yaml example
    # runs:
    #   - <dark_fast prepared csv>
    #   - <bright_fast prepared csv>
    # (val)
    #   - <dark_slow prepared csv>
    #   - <bright_slow prepared csv>

if __name__ == "__main__":
    main()
