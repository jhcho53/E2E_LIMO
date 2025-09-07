# rgb_lane_train_eval.py
from __future__ import annotations
import os, math, json, yaml, argparse, time, random
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_bin_centers(vmin: float, vmax: float, K: int) -> np.ndarray:
    edges = np.linspace(vmin, vmax, K + 1, dtype=np.float32)
    return 0.5 * (edges[:-1] + edges[1:])  # (K,)

def soft_label_scalar(x: float, centers: np.ndarray, sigma: float) -> np.ndarray:
    d = (centers - x) / (sigma + 1e-9)
    w = np.exp(-0.5 * d * d)
    w = w / (w.sum() + 1e-9)
    return w.astype(np.float32)

def bin_index_from_values(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # nearest-center index
    idx = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
    return idx

# =========================
# Dataset (status 없이) + 견고한 경로 처리
# =========================
class LaneControlSeqDataset(Dataset):
    """
    CSV 포맷(필수 열):
      file, stamp_ns, twist_linear_x_mps, twist_angular_z_rps
    * 이미지 시퀀스(T=seq_len)와 마지막 시점 라벨 사용
    * 경로 처리 우선순위: 절대경로 > img_root+상대경로 > CSV폴더+상대경로
    * drop_missing_images=True면 존재하지 않는 이미지 행을 제거
    """
    COL_FILE = "file"
    COL_STAMP = "stamp_ns"
    COL_V = "twist_linear_x_mps"
    COL_W = "twist_angular_z_rps"

    def __init__(
        self,
        list_yaml: str,
        seq_len: int = 8,
        stride: int = 1,
        img_size: int = 224,
        train: bool = True,
        use_class_label: bool = True,
        v_bin_cfg: Tuple[float, float, int] = (0.0, 8.0, 21),
        w_bin_cfg: Tuple[float, float, int] = (-0.6, 0.6, 41),
        v_soft_sigma: float = 0.4,
        w_soft_sigma: float = 0.03,
        drop_missing_images: bool = True,
        img_root: str = "",      # ✅ 이미지 루트(선택)
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.train = train
        self.use_class_label = use_class_label
        self.drop_missing = drop_missing_images
        self.img_root = img_root.strip()

        self.vmin, self.vmax, self.Kv = v_bin_cfg
        self.wmin, self.wmax, self.Kw = w_bin_cfg
        self.v_centers = make_bin_centers(self.vmin, self.vmax, self.Kv)
        self.w_centers = make_bin_centers(self.wmin, self.wmax, self.Kw)
        self.v_soft_sigma = v_soft_sigma
        self.w_soft_sigma = w_soft_sigma

        aug = []
        if train:
            aug += [transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                           saturation=0.1, hue=0.05)]
        aug += [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ]
        self.img_tf = transforms.Compose(aug)

        with open(list_yaml, "r") as f:
            runs = yaml.safe_load(f)["runs"]

        self.tables: Dict[str, pd.DataFrame] = {}
        self.samples: List[Tuple[str,int]] = []

        for csv_path in runs:
            df = pd.read_csv(csv_path)
            if self.COL_STAMP in df.columns:
                df = df.sort_values(self.COL_STAMP).reset_index(drop=True)

            df = self._resolve_paths(df, csv_path)  # ✅ 핵심

            if len(df) < seq_len:
                print(f"[WARN] {csv_path}: too short after filtering (rows={len(df)}), skip")
                continue

            self.tables[csv_path] = df
            last_start = len(df) - seq_len
            for st in range(0, last_start + 1, self.stride):
                self.samples.append((csv_path, st))

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check YAML/CSV paths, 'file' column, and --img_root.")

    def _resolve_paths(self, df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
        if self.COL_FILE not in df.columns:
            raise ValueError(f"{csv_path}에 '{self.COL_FILE}' 컬럼이 없습니다.")

        base_csv = os.path.dirname(csv_path)
        root = self.img_root  # 비어있을 수 있음

        def to_abs(p: str):
            if not isinstance(p, str):
                return p
            if os.path.isabs(p):
                return p
            if root:
                return os.path.normpath(os.path.join(root, p))
            return os.path.normpath(os.path.join(base_csv, p))

        df = df.copy()
        df[self.COL_FILE] = df[self.COL_FILE].apply(to_abs)

        if self.drop_missing:
            exists_mask = df[self.COL_FILE].apply(os.path.exists)
            missing = (~exists_mask).sum()
            if missing > 0:
                print(f"[INFO] {csv_path}: drop {missing} rows (missing image files)")
            df = df[exists_mask].reset_index(drop=True)
        return df

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.img_tf(img)

    def __getitem__(self, idx: int):
        csv_path, st = self.samples[idx]
        df = self.tables[csv_path]
        ed = st + self.seq_len
        win = df.iloc[st:ed].reset_index(drop=True)

        # 이미지 시퀀스
        imgs = [self._load_img(p) for p in win[self.COL_FILE].tolist()]
        imgs = torch.stack(imgs, dim=0)  # [T,3,H,W]

        # 라벨(마지막 시점)
        last = win.iloc[-1]
        v_gt = float(last[self.COL_V])
        w_gt = float(last[self.COL_W])

        if self.use_class_label:
            v_soft = soft_label_scalar(v_gt, self.v_centers, self.v_soft_sigma)
            w_soft = soft_label_scalar(w_gt, self.w_centers, self.w_soft_sigma)
            label = {
                "v_soft": torch.from_numpy(v_soft),
                "w_soft": torch.from_numpy(w_soft),
            }
        else:
            label = {
                "v": torch.tensor([v_gt], dtype=torch.float32),
                "w": torch.tensor([w_gt], dtype=torch.float32),
            }

        meta = {
            "csv_path": csv_path,
            "start_index": st,
            "stamp_last": int(last[self.COL_STAMP]) if self.COL_STAMP in win.columns else -1,
            "v_gt": v_gt,
            "w_gt": w_gt,
        }
        return {"images": imgs, "label": label, "meta": meta}

# =========================
# Model (ResNet18 + GRU + MLP heads)
# =========================
class RGB2CtrlNoStatus(nn.Module):
    def __init__(self, K_v=21, K_w=41, hidden=256, backbone_out=512):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # -> 512
        self.gru = nn.GRU(input_size=backbone_out, hidden_size=hidden,
                          num_layers=1, batch_first=True)
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.head_v = nn.Linear(128, K_v)
        self.head_w = nn.Linear(128, K_w)

    def forward(self, x_btchw):
        B,T,C,H,W = x_btchw.shape
        x = x_btchw.reshape(B*T, C, H, W)
        feat = self.backbone(x)           # (B*T,512)
        feat = feat.view(B, T, -1)        # (B,T,512)
        o,_ = self.gru(feat)              # (B,T,hidden)
        o_t = o[:, -1, :]                 # (B,hidden)
        z = self.trunk(o_t)               # (B,128)
        logits_v = self.head_v(z)         # (B,K_v)
        logits_w = self.head_w(z)         # (B,K_w)
        return logits_v, logits_w

# =========================
# Train / Eval
# =========================
def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             v_centers: np.ndarray,
             w_centers: np.ndarray,
             device: torch.device):
    model.eval()
    total = 0
    sum_loss = 0.0

    v_mae_exp, w_mae_exp = 0.0, 0.0
    v_top1_mae, w_top1_mae = 0.0, 0.0
    v_top1_acc, w_top1_acc = 0.0, 0.0

    v_cent = torch.tensor(v_centers, device=device)
    w_cent = torch.tensor(w_centers, device=device)

    for batch in loader:
        x = batch["images"].to(device)    # [B,T,3,H,W]
        v_gt = torch.tensor(batch["meta"]["v_gt"], dtype=torch.float32, device=device)
        w_gt = torch.tensor(batch["meta"]["w_gt"], dtype=torch.float32, device=device)

        logits_v, logits_w = model(x)

        if "v_soft" in batch["label"]:
            v_soft = batch["label"]["v_soft"].to(device)
            w_soft = batch["label"]["w_soft"].to(device)
            loss = soft_ce_loss(logits_v, v_soft) + soft_ce_loss(logits_w, w_soft)
            sum_loss += loss.item() * x.size(0)

        pv = F.softmax(logits_v, dim=-1)
        pw = F.softmax(logits_w, dim=-1)
        v_hat = (pv * v_cent).sum(dim=-1)  # (B,)
        w_hat = (pw * w_cent).sum(dim=-1)  # (B,)
        v_mae_exp += (v_hat - v_gt).abs().sum().item()
        w_mae_exp += (w_hat - w_gt).abs().sum().item()

        v_idx = logits_v.argmax(dim=-1)
        w_idx = logits_w.argmax(dim=-1)
        v_hat_top1 = v_cent[v_idx]
        w_hat_top1 = w_cent[w_idx]
        v_top1_mae += (v_hat_top1 - v_gt).abs().sum().item()
        w_top1_mae += (w_hat_top1 - w_gt).abs().sum().item()
        v_top1_acc += (v_idx.cpu().numpy() == bin_index_from_values(v_gt.cpu().numpy(), v_centers)).sum().item()
        w_top1_acc += (w_idx.cpu().numpy() == bin_index_from_values(w_gt.cpu().numpy(), w_centers)).sum().item()

        total += x.size(0)

    res = {
        "loss": (sum_loss / total) if total > 0 else float("nan"),
        "v_mae_exp": v_mae_exp / total,
        "w_mae_exp": w_mae_exp / total,
        "v_mae_top1": v_top1_mae / total,
        "w_mae_top1": w_top1_mae / total,
        "v_top1_acc": v_top1_acc / total,
        "w_top1_acc": w_top1_acc / total,
    }
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_yaml", type=str, required=True)
    p.add_argument("--val_yaml",   type=str, required=True)
    p.add_argument("--out_dir",    type=str, default="./runs/exp1")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")

    # bin config
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--Kv",   type=int,   default=21)
    p.add_argument("--wmin", type=float, default=-0.6)
    p.add_argument("--wmax", type=float, default= 0.6)
    p.add_argument("--Kw",   type=int,   default=41)
    p.add_argument("--v_sigma", type=float, default=0.4)
    p.add_argument("--w_sigma", type=float, default=0.03)

    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--img_root", type=str, default="", help="file이 상대경로일 때 붙일 최상위 이미지 폴더")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    v_centers = make_bin_centers(args.vmin, args.vmax, args.Kv)
    w_centers = make_bin_centers(args.wmin, args.wmax, args.Kw)

    # Datasets / Loaders
    train_ds = LaneControlSeqDataset(
        args.train_yaml,
        seq_len=args.seq_len,
        stride=args.stride,
        img_size=args.img_size,
        train=True,
        use_class_label=True,
        v_bin_cfg=(args.vmin, args.vmax, args.Kv),
        w_bin_cfg=(args.wmin, args.wmax, args.Kw),
        v_soft_sigma=args.v_sigma,
        w_soft_sigma=args.w_sigma,
        drop_missing_images=True,
        img_root=args.img_root,
    )
    val_ds = LaneControlSeqDataset(
        args.val_yaml,
        seq_len=args.seq_len,
        stride=args.stride,
        img_size=args.img_size,
        train=False,
        use_class_label=True,
        v_bin_cfg=(args.vmin, args.vmax, args.Kv),
        w_bin_cfg=(args.wmin, args.wmax, args.Kw),
        v_soft_sigma=args.v_sigma,
        w_soft_sigma=args.w_sigma,
        drop_missing_images=True,
        img_root=args.img_root,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"SeqLen={args.seq_len}, ImgSize={args.img_size}, Kv={args.Kv}, Kw={args.Kw}")

    # Model / Optim
    model = RGB2CtrlNoStatus(K_v=args.Kv, K_w=args.Kw, hidden=256).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # (PyTorch 2.1+ 경고 대응) amp 스케일러
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Resume
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    def save_ckpt(epoch: int, best: bool=False):
        path = os.path.join(args.out_dir, f"{'best' if best else f'ep{epoch:03d}'}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": scheduler.state_dict(),
            "config": vars(args),
        }, path)
        print(f"Saved checkpoint: {path}")

    # Training loop
    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        t0 = time.time()
        for batch in train_loader:
            x = batch["images"].to(device)
            v_soft = batch["label"]["v_soft"].to(device)
            w_soft = batch["label"]["w_soft"].to(device)

            with torch.amp.autocast('cuda', enabled=args.amp):
                logits_v, logits_w = model(x)
                loss = soft_ce_loss(logits_v, v_soft) + soft_ce_loss(logits_w, w_soft)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            epoch_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        scheduler.step()
        tr_loss = epoch_loss / max(1, n_samples)

        # Evaluate
        val_res = evaluate(model, val_loader, v_centers, w_centers, device)

        t1 = time.time()
        print(f"[{epoch:02d}] "
              f"train_loss={tr_loss:.4f}  "
              f"val_loss={val_res['loss']:.4f}  "
              f"v_MAE_exp={val_res['v_mae_exp']:.4f}  w_MAE_exp={val_res['w_mae_exp']:.4f}  "
              f"v_Top1Acc={val_res['v_top1_acc']:.3f}  w_Top1Acc={val_res['w_top1_acc']:.3f}  "
              f"({t1-t0:.1f}s)")

        # Save best by expectation MAE sum
        score = val_res["v_mae_exp"] + val_res["w_mae_exp"]
        if score < best_val:
            best_val = score
            save_ckpt(epoch, best=True)

        if (epoch % args.save_every) == 0:
            save_ckpt(epoch, best=False)

    print("Done.")

if __name__ == "__main__":
    main()
