#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, time, math
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
from torchvision.models import resnet18, ResNet18_Weights

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


# ---------------- Model ----------------
class RGB2CtrlNoStatus(nn.Module):
    def __init__(self, K_v=21, K_w=41, hidden=256, backbone_out=512):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
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
        feat = self.backbone(x)
        feat = feat.view(B, T, -1)
        o,_ = self.gru(feat)
        z = self.trunk(o[:, -1, :])
        return self.head_v(z), self.head_w(z)


# -------------- Utils --------------
def make_bin_centers(vmin: float, vmax: float, K: int):
    edges = np.linspace(vmin, vmax, K + 1, dtype=np.float32)
    return torch.tensor(0.5 * (edges[:-1] + edges[1:]), dtype=torch.float32)


# -------------- Node --------------
class FastE2ENode(Node):
    def __init__(self, args):
        super().__init__("e2e_rgb2cmdvel_fast")

        # Device & FP16
        self.device = torch.device(args.device)
        self.use_amp = (self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Load ckpt
        ckpt = torch.load(args.checkpoint, map_location=self.device)
        cfg = ckpt.get("config", {})

        # Binning (ckpt 우선, 없으면 CLI)
        self.vmin = float(cfg.get("vmin", args.vmin))
        self.vmax = float(cfg.get("vmax", args.vmax))
        self.Kv   = int(cfg.get("Kv",   args.Kv))
        self.wmin = float(cfg.get("wmin", args.wmin))
        self.wmax = float(cfg.get("wmax", args.wmax))
        self.Kw   = int(cfg.get("Kw",   args.Kw))

        # ★ seq_len / img_size 는 항상 CLI 우선
        self.seq_len  = int(args.seq_len)
        self.img_size = int(args.img_size)

        # Model
        self.model = RGB2CtrlNoStatus(K_v=self.Kv, K_w=self.Kw, hidden=256).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        if self.use_amp:
            self.model = self.model.half()  # channels_last 미사용(안전)

        # Bin centers
        self.v_centers = make_bin_centers(self.vmin, self.vmax, self.Kv).to(self.device)
        self.w_centers = make_bin_centers(self.wmin, self.wmax, self.Kw).to(self.device)
        if self.use_amp:
            self.v_centers = self.v_centers.half()
            self.w_centers = self.w_centers.half()

        # Normalize params
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device)
        if self.use_amp:
            self.mean = self.mean.half()
            self.std  = self.std.half()

        # 최신 프레임만 유지 + 시퀀스
        self.latest_tensor: Optional[torch.Tensor] = None
        self.window = deque(maxlen=self.seq_len)

        # ---------- Runtime parameters ----------
        self.expectation_output = args.expectation_output
        self.min_pub_interval   = args.min_pub_interval
        self.speed_scale        = args.speed_scale
        self.turn_scale         = args.turn_scale
        self.v_clip = tuple(args.v_clip) if args.v_clip else None
        self.w_clip = tuple(args.w_clip) if args.w_clip else None

        # ---------- Smoothing / Safety knobs ----------
        self.ema_v = float(args.ema_v)         # 0(끄기)~1(즉시반응). 0.2~0.5 권장
        self.ema_w = float(args.ema_w)
        self.max_accel      = float(args.max_accel)      # m/s^2
        self.max_ang_accel  = float(args.max_ang_accel)  # rad/s^2
        self.curve_k        = float(args.curve_k)        # 커브 감속 강도 0~1+
        self.conf_smooth    = float(args.conf_smooth)    # 저신뢰시 추가 EMA 강도
        self.conf_min       = float(args.conf_min)       # 이 값 미만 신뢰도면 완화 적용

        # State for filters
        self.prev_v = 0.0
        self.prev_w = 0.0
        self.prev_time = time.time()

        # ROS I/O
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,   # 최신 프레임 우선
        )
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, args.image_topic, self.image_cb, qos)
        self.pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)

        # Timer
        period = max(self.min_pub_interval, 1e-3)
        self.timer = self.create_timer(period, self.infer_and_publish)
        self._last_pub = 0.0

        self.get_logger().info(
            f"FAST node | seq_len={self.seq_len}, img={self.img_size}, device={self.device}, "
            f"Kv={self.Kv}[{self.vmin},{self.vmax}], Kw={self.Kw}[{self.wmin},{self.wmax}] | "
            f"EMA(v={self.ema_v}, w={self.ema_w}), accel(limit={self.max_accel},{self.max_ang_accel}), curve_k={self.curve_k}"
        )

    def image_cb(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return

        if (cv_bgr.shape[1], cv_bgr.shape[0]) != (self.img_size, self.img_size):
            cv_bgr = cv2.resize(cv_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)

        t = torch.from_numpy(cv_rgb).to(self.device)
        t = t.permute(2,0,1).contiguous()  # HWC→CHW
        t = (t.half() if self.use_amp else t.float()) / 255.0
        t = (t - self.mean[:, None, None]) / self.std[:, None, None]
        self.latest_tensor = t

    @torch.no_grad()
    def infer_and_publish(self):
        if self.latest_tensor is None:
            return

        self.window.append(self.latest_tensor)
        if len(self.window) < self.seq_len:
            return

        now = time.time()
        dt = max(1e-3, now - self.prev_time)
        if (now - self._last_pub) < self.min_pub_interval:
            return

        frames = torch.stack(list(self.window), dim=0).unsqueeze(0)  # [1,T,3,H,W]

        # 모델 추론
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_v, logits_w = self.model(frames)
        else:
            logits_v, logits_w = self.model(frames)

        # softmax + 신뢰도(엔트로피)
        pv = F.softmax(logits_v, dim=-1)
        pw = F.softmax(logits_w, dim=-1)

        if self.expectation_output:
            v_raw = (pv * self.v_centers).sum(dim=-1).item()
            w_raw = (pw * self.w_centers).sum(dim=-1).item()
        else:
            v_raw = self.v_centers[logits_v.argmax(dim=-1)].item()
            w_raw = self.w_centers[logits_w.argmax(dim=-1)].item()

        # 스케일
        v_cmd = v_raw * self.speed_scale
        w_cmd = w_raw * self.turn_scale

        # ---- 커브 감속 ----
        curve_scale = 1.0 / (1.0 + self.curve_k * (abs(w_cmd) / max(1e-6, self.wmax)))
        v_cmd *= float(np.clip(curve_scale, 0.2, 1.0))  # 과도 감속 방지 하한

        # ---- EMA ----
        if self.ema_v > 0.0:
            v_cmd = (1 - self.ema_v) * self.prev_v + self.ema_v * v_cmd
        if self.ema_w > 0.0:
            w_cmd = (1 - self.ema_w) * self.prev_w + self.ema_w * w_cmd

        # ---- 신뢰도 기반 완화 ----
        def norm_entropy(p):
            p = torch.clamp(p.squeeze(0), 1e-8, 1.0)
            H = -(p * torch.log(p)).sum()
            Hmax = math.log(p.numel())
            return float(torch.clamp(H / Hmax, 0.0, 1.0))

        nv = norm_entropy(pv)
        nw = norm_entropy(pw)
        conf = 1.0 - 0.5 * (nv + nw)  # 0~1 (1=높은 확신)
        if conf < self.conf_min:
            alpha = self.conf_smooth  # 0~1
            v_cmd = (1 - alpha) * self.prev_v + alpha * v_cmd
            w_cmd = (1 - alpha) * self.prev_w + alpha * w_cmd
            v_cmd *= 0.8  # 저신뢰시 속도 살짝 낮춤

        # ---- Slew-rate(가속) 제한 ----
        dv_max = self.max_accel * dt
        dw_max = self.max_ang_accel * dt
        v_cmd = float(np.clip(v_cmd, self.prev_v - dv_max, self.prev_v + dv_max))
        w_cmd = float(np.clip(w_cmd, self.prev_w - dw_max, self.prev_w + dw_max))

        # 최종 클램프
        if self.v_clip: v_cmd = float(np.clip(v_cmd, self.v_clip[0], self.v_clip[1]))
        if self.w_clip: w_cmd = float(np.clip(w_cmd, self.w_clip[0], self.w_clip[1]))

        # Publish
        twist = Twist()
        twist.linear.x  = v_cmd
        twist.angular.z = w_cmd
        self.pub.publish(twist)

        # 상태 갱신
        self.prev_v = v_cmd
        self.prev_w = w_cmd
        self.prev_time = now
        self._last_pub = now


def parse_args():
    p = argparse.ArgumentParser("E2E RGB->cmd_vel (FAST, CLI-priority)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image_topic", type=str, default="/camera/image_raw")
    p.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # ★ CLI 우선(ckpt에 값 있어도 무시)
    p.add_argument("--seq_len", type=int, default=4)
    p.add_argument("--img_size", type=int, default=160)

    # binning은 ckpt 우선, 없으면 아래 사용
    p.add_argument("--Kv", type=int, default=21)
    p.add_argument("--Kw", type=int, default=41)
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--wmin", type=float, default=-0.6)
    p.add_argument("--wmax", type=float, default=0.6)

    # 출력/퍼블리시
    p.add_argument("--expectation_output", action="store_true")
    p.add_argument("--min_pub_interval", type=float, default=0.02)
    p.add_argument("--speed_scale", type=float, default=1.0)
    p.add_argument("--turn_scale", type=float, default=1.0)
    p.add_argument("--v_clip", type=float, nargs=2, default=None)
    p.add_argument("--w_clip", type=float, nargs=2, default=None)

    # ★ 새로 추가된 스무딩/안전 파라미터들
    p.add_argument("--ema_v", type=float, default=0.3)
    p.add_argument("--ema_w", type=float, default=0.4)
    p.add_argument("--max_accel", type=float, default=0.6)        # m/s^2
    p.add_argument("--max_ang_accel", type=float, default=2.5)     # rad/s^2
    p.add_argument("--curve_k", type=float, default=0.6)
    p.add_argument("--conf_smooth", type=float, default=0.5)
    p.add_argument("--conf_min", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = FastE2ENode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
