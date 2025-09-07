#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, argparse
from collections import deque
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image as PILImage
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


# -------------------------
# Model (학습 스크립트와 동일)
# -------------------------
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


# -------------------------
# Utils
# -------------------------
def make_bin_centers(vmin: float, vmax: float, K: int) -> np.ndarray:
    edges = np.linspace(vmin, vmax, K + 1, dtype=np.float32)
    return 0.5 * (edges[:-1] + edges[1:])

def build_transform(img_size: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])


# -------------------------
# ROS2 Node
# -------------------------
class E2ERGB2CmdVel(Node):
    def __init__(self, args):
        super().__init__("e2e_rgb2cmdvel")

        # ---- Load checkpoint & config
        self.device = torch.device(args.device)
        ckpt = torch.load(args.checkpoint, map_location=self.device)
        cfg = ckpt.get("config", {})  # 학습 args가 여기에 저장되어 있음

        # bin / 해상도 / 시퀀스 설정 (ckpt 우선, 없으면 CLI 기본)
        self.vmin = float(cfg.get("vmin", args.vmin))
        self.vmax = float(cfg.get("vmax", args.vmax))
        self.Kv   = int(cfg.get("Kv",   args.Kv))
        self.wmin = float(cfg.get("wmin", args.wmin))
        self.wmax = float(cfg.get("wmax", args.wmax))
        self.Kw   = int(cfg.get("Kw",   args.Kw))

        self.seq_len = int(cfg.get("seq_len", args.seq_len))
        self.img_size = int(cfg.get("img_size", args.img_size))

        # 모델/파이프라인
        self.model = RGB2CtrlNoStatus(K_v=self.Kv, K_w=self.Kw, hidden=256).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.v_centers = torch.tensor(make_bin_centers(self.vmin, self.vmax, self.Kv), device=self.device)
        self.w_centers = torch.tensor(make_bin_centers(self.wmin, self.wmax, self.Kw), device=self.device)

        self.tf = build_transform(self.img_size)
        self.bridge = CvBridge()
        self.buffer: Deque[torch.Tensor] = deque(maxlen=self.seq_len)

        # 런타임 파라미터
        self.expectation_output = args.expectation_output
        self.min_pub_interval = args.min_pub_interval
        self.speed_scale = args.speed_scale
        self.turn_scale = args.turn_scale
        self.v_clip = tuple(args.v_clip) if args.v_clip else None  # (min,max)
        self.w_clip = tuple(args.w_clip) if args.w_clip else None  # (min,max)

        # ROS I/O
        qos_sensors = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub = self.create_subscription(Image, args.image_topic, self.image_cb, qos_sensors)
        self.pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)

        self._last_pub = 0.0
        self.get_logger().info(
            f"ckpt={args.checkpoint} | device={self.device} | seq_len={self.seq_len} | img={self.img_size} "
            f"| Kv={self.Kv} [{self.vmin},{self.vmax}] | Kw={self.Kw} [{self.wmin},{self.wmax}]"
        )

    def image_cb(self, msg: Image):
        # BGR8 -> PIL RGB
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return
        pil = PILImage.fromarray(cv_img[:, :, ::-1])  # BGR -> RGB

        x = self.tf(pil)  # [3,H,W]
        self.buffer.append(x)

        if len(self.buffer) < self.seq_len:
            return

        now = time.time()
        if (now - self._last_pub) < self.min_pub_interval:
            return

        frames = torch.stack(list(self.buffer), dim=0).unsqueeze(0).to(self.device)  # [1,T,3,H,W]
        with torch.no_grad():
            logits_v, logits_w = self.model(frames)
            if self.expectation_output:
                pv = F.softmax(logits_v, dim=-1)
                pw = F.softmax(logits_w, dim=-1)
                v = (pv * self.v_centers).sum(dim=-1)  # [1]
                w = (pw * self.w_centers).sum(dim=-1)
            else:
                v = self.v_centers[logits_v.argmax(dim=-1)]  # [1]
                w = self.w_centers[logits_w.argmax(dim=-1)]

            v = float(v.item() * self.speed_scale)
            w = float(w.item() * self.turn_scale)

        # 안전 클램프(옵션)
        if self.v_clip:
            v = float(np.clip(v, self.v_clip[0], self.v_clip[1]))
        if self.w_clip:
            w = float(np.clip(w, self.w_clip[0], self.w_clip[1]))

        twist = Twist()
        twist.linear.x  = v
        twist.angular.z = w
        self.pub.publish(twist)
        self._last_pub = now


# -------------------------
# main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="E2E RGB->cmd_vel (ResNet18+GRU)")
    p.add_argument("--checkpoint", type=str, required=True, help="학습 .pth (best.pth)")
    p.add_argument("--image_topic", type=str, default="/camera/image_raw")
    p.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 추론 파라미터(ckpt에 없으면 사용)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--Kv", type=int, default=21)
    p.add_argument("--Kw", type=int, default=41)
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=8.0)
    p.add_argument("--wmin", type=float, default=-0.6)
    p.add_argument("--wmax", type=float, default= 0.6)

    # 출력/퍼블리시 옵션
    p.add_argument("--expectation_output", action="store_true", help="softmax 기대값 사용(기본: top-1)")
    p.add_argument("--min_pub_interval", type=float, default=0.05, help="퍼블리시 최소 간격(초)")
    p.add_argument("--speed_scale", type=float, default=1.0, help="선속도 스케일")
    p.add_argument("--turn_scale", type=float, default=1.0, help="각속도 스케일")
    p.add_argument("--v_clip", type=float, nargs=2, default=None, help="선속도 클램프 eg. 0.0 1.0")
    p.add_argument("--w_clip", type=float, nargs=2, default=None, help="각속도 클램프 eg. -1.0 1.0")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = E2ERGB2CmdVel(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
