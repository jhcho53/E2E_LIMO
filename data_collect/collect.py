#!/usr/bin/env python3
# live_e2e_recorder_twist_nocrop.py
# Real-time E2E dataset recorder (Twist-only control; NO crop, NO resize)
# - Subscribes: front camera (Image or CompressedImage)
# - Subscribes: /cmd_vel (Twist or TwistStamped)  --> labels: linear.x [m/s], angular.z [rad/s]
# - Optional:   /odom (nav_msgs/Odometry)         --> executed v/ω (aux columns)
# - Optional:   /imu  (sensor_msgs/Imu)           --> gyro_z, accel_x (aux columns)
# - Optional:   /camera_info (CameraInfo)         --> save to camera_info.yaml once
# - Aligns by timestamp with configurable delay: image time + delay -> nearest control sample within tolerance
# - Saves: images/ + labels.csv (one row per saved image)
#
# Usage example:
#   python3 live_e2e_recorder_twist_nocrop.py \
#     --ros-args \
#     -p image_topic:=/camera/front/image_raw \
#     -p image_is_compressed:=false \
#     -p twist_topic:=/cmd_vel \
#     -p twist_is_stamped:=false \
#     -p odom_topic:=/odom \
#     -p imu_topic:=/imu/data \
#     -p camera_info_topic:=/camera/front/camera_info \
#     -p out_dir:=/tmp/e2e_dataset_live \
#     -p delay_ms:=120.0 \
#     -p align_tol_ms:=80.0 \
#     -p a_max:=1.5 -p b_max:=2.0 \
#     -p save_fps_max:=15.0
#
# Dependencies:
#   pip install opencv-python-headless numpy pandas pyyaml
#
import os, sys, math, csv, yaml
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Deque

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time as RosTime

from sensor_msgs.msg import Image, CompressedImage, Imu, CameraInfo
from geometry_msgs.msg import Twist
try:
    from geometry_msgs.msg import TwistStamped
    HAVE_TWIST_STAMPED = True
except Exception:
    HAVE_TWIST_STAMPED = False

from nav_msgs.msg import Odometry

def ros_time_to_ns(t: RosTime) -> int:
    return int(t.sec) * 1_000_000_000 + int(t.nanosec)

def now_ns(node: Node) -> int:
    return int(node.get_clock().now().nanoseconds)

def get_msg_stamp_ns(node: Node, msg) -> int:
    stamp = getattr(msg, "header", None)
    if stamp is not None and hasattr(stamp, "stamp"):
        return ros_time_to_ns(stamp.stamp)
    return now_ns(node)

def to_bgr_image_from_raw(msg: Image) -> np.ndarray:
    h = int(msg.height); w = int(msg.width); enc = str(msg.encoding).lower()
    step = int(msg.step)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("bgr8", "rgb8"):
        ch = 3; row_bytes = step
        if row_bytes != w * ch:
            img = buf.reshape(h, row_bytes)[:, :w*ch].reshape(h, w, ch)
        else:
            img = buf.reshape(h, w, ch)
        if enc == "rgb8":
            img = img[:, :, ::-1]
        return img
    elif enc in ("bgra8", "rgba8"):
        ch = 4; row_bytes = step
        if row_bytes != w * ch:
            img = buf.reshape(h, row_bytes)[:, :w*ch].reshape(h, w, ch)
        else:
            img = buf.reshape(h, w, ch)
        if enc == "rgba8":
            img = img[:, :, [2,1,0,3]]
        return img[:, :, :3]
    elif enc in ("mono8",):
        row_bytes = step
        if row_bytes != w:
            img = buf.reshape(h, row_bytes)[:, :w]
        else:
            img = buf.reshape(h, w)
        return img
    else:
        raise RuntimeError(f"Unsupported Image encoding: {msg.encoding}")

def to_bgr_image_from_compressed(msg: CompressedImage) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    return img

class TimeSeries:
    """Simple time-indexed series with nearest lookup and acceleration from v(t)."""
    def __init__(self, maxlen: int = 20000):
        self.t: Deque[int] = deque(maxlen=maxlen)
        self.v: Deque[dict] = deque(maxlen=maxlen)
    def append(self, t_ns: int, val: dict):
        self.t.append(int(t_ns)); self.v.append(val)
    def nearest(self, t_ns: int, tol_ns: int) -> Optional[Tuple[dict, int]]:
        if not self.t: return None
        ts = list(self.t); vs = list(self.v)
        import bisect
        i = bisect.bisect_left(ts, t_ns)
        cand = []
        if i < len(ts): cand.append((abs(ts[i]-t_ns), i))
        if i > 0:        cand.append((abs(ts[i-1]-t_ns), i-1))
        if not cand: return None
        d, j = min(cand, key=lambda x: x[0])
        if d <= tol_ns:
            return vs[j], ts[j]
        return None
    def accel_from_v(self, t_ns: int, key: str = "v") -> Optional[float]:
        """Estimate dv/dt around t_ns (m/s^2). Uses value 'key' from timeseries entries."""
        n = len(self.t)
        if n < 2: return None
        ts = list(self.t); vs = list(self.v)
        import bisect
        i = bisect.bisect_left(ts, t_ns)
        if i <= 0: j0, j1 = 0, 1
        elif i >= n: j0, j1 = n-2, n-1
        else:
            if i+1 < n and i-1 >= 0: j0, j1 = i-1, i+1
            else:                    j0, j1 = i-1, i
        dt = ts[j1] - ts[j0]
        if dt <= 0: return None
        v0 = float(vs[j0].get(key, 0.0)); v1 = float(vs[j1].get(key, 0.0))
        return (v1 - v0) / (dt * 1e-9)

class LiveTwistOnlyRecorder(Node):
    def __init__(self):
        super().__init__("live_e2e_recorder_twist_nocrop")
        # Parameters (NO crop; NO resize)
        self.declare_parameter("image_topic", "/camera/front/image_raw")
        self.declare_parameter("image_is_compressed", False)
        self.declare_parameter("twist_topic", "/cmd_vel")
        self.declare_parameter("twist_is_stamped", False)
        self.declare_parameter("odom_topic", "")           # optional
        self.declare_parameter("imu_topic", "")            # optional
        self.declare_parameter("camera_info_topic", "")    # optional
        self.declare_parameter("out_dir", "/tmp/e2e_dataset_live")
        self.declare_parameter("delay_ms", 120.0)
        self.declare_parameter("align_tol_ms", 80.0)
        self.declare_parameter("a_max", 1.5)
        self.declare_parameter("b_max", 2.0)
        self.declare_parameter("save_fps_max", 0.0)       # 0=unlimited
        self.declare_parameter("image_buffer_sec", 5.0)    # buffer time for images
        self.declare_parameter("process_hz", 30.0)         # processing rate
        self.declare_parameter("csv_flush_every", 50)      # flush rows every N saves

        # Load params
        p = self.get_parameters([
            "image_topic","image_is_compressed","twist_topic","twist_is_stamped",
            "odom_topic","imu_topic","camera_info_topic","out_dir",
            "delay_ms","align_tol_ms","a_max","b_max",
            "save_fps_max","image_buffer_sec","process_hz","csv_flush_every"
        ])
        params = {pp.name: pp.value for pp in p}

        self.out_dir = Path(params["out_dir"]); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir = self.out_dir / "images"; self.img_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "labels.csv"
        self.meta_path = self.out_dir / "meta_live.json"

        self.delay_ns = int(float(params["delay_ms"]) * 1e6)
        self.tol_ns   = int(float(params["align_tol_ms"]) * 1e6)
        self.a_max    = float(params["a_max"]); self.b_max = float(params["b_max"])
        self.save_fps_max = float(params["save_fps_max"])
        self.image_buffer_sec = float(params["image_buffer_sec"])
        self.csv_flush_every = int(params["csv_flush_every"])
        self.twist_is_stamped = bool(params["twist_is_stamped"])

        # Buffers
        self.images = deque()       # (stamp_ns, fname, img_bgr)
        self.twist_ts = TimeSeries()
        self.odom_ts  = TimeSeries() if params["odom_topic"] else None
        self.imu_ts   = TimeSeries() if params["imu_topic"] else None

        # Counters
        self.saved = 0; self.seen = 0; self.skipped_align = 0; self.last_save_ns = 0

        # CSV init
        self.csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self.csv_writer = None; self.csv_header_written = False; self.csv_rows_since_flush = 0

        # QoS
        qos_img = QoSProfile(depth=10); qos_img.reliability = ReliabilityPolicy.BEST_EFFORT; qos_img.history = HistoryPolicy.KEEP_LAST
        qos_ctrl = QoSProfile(depth=50); qos_ctrl.reliability = ReliabilityPolicy.RELIABLE; qos_ctrl.history = HistoryPolicy.KEEP_LAST

        # Subscriptions
        img_topic = params["image_topic"]
        if bool(params["image_is_compressed"]):
            self.image_sub = self.create_subscription(CompressedImage, img_topic, self.on_compressed_image, qos_img)
            self.get_logger().info(f"Subscribed to CompressedImage: {img_topic}")
        else:
            self.image_sub = self.create_subscription(Image, img_topic, self.on_image, qos_img)
            self.get_logger().info(f"Subscribed to Image: {img_topic}")

        if self.twist_is_stamped and HAVE_TWIST_STAMPED:
            self.twist_sub = self.create_subscription(TwistStamped, params["twist_topic"], self.on_twist_stamped, qos_ctrl)
            self.get_logger().info(f"Subscribed to TwistStamped: {params['twist_topic']}")
        else:
            self.twist_sub = self.create_subscription(Twist, params["twist_topic"], self.on_twist, qos_ctrl)
            self.get_logger().info(f"Subscribed to Twist: {params['twist_topic']}")

        if params["odom_topic"]:
            self.odom_sub = self.create_subscription(Odometry, params["odom_topic"], self.on_odom, qos_ctrl)
            self.get_logger().info(f"Subscribed to Odometry: {params['odom_topic']}")
        else:
            self.odom_sub = None

        if params["imu_topic"]:
            self.imu_sub = self.create_subscription(Imu, params["imu_topic"], self.on_imu, qos_ctrl)
            self.get_logger().info(f"Subscribed to Imu: {params['imu_topic']}")
        else:
            self.imu_sub = None

        # CameraInfo (optional, save once)
        self.cam_info_saved = False
        if params["camera_info_topic"]:
            self.cinfo_sub = self.create_subscription(CameraInfo, params["camera_info_topic"], self.on_camera_info, qos_ctrl)
            self.get_logger().info(f"Subscribed to CameraInfo: {params['camera_info_topic']}")
        else:
            self.cinfo_sub = None

        # Processing timer
        hz = float(params["process_hz"]); period = 1.0 / max(1e-3, hz)
        self.timer = self.create_timer(period, self.process_buffers)

        # Save meta
        meta = {
            "out_dir": str(self.out_dir),
            "delay_ms": float(params["delay_ms"]),
            "align_tol_ms": float(params["align_tol_ms"]),
            "image_topic": img_topic, "image_is_compressed": bool(params["image_is_compressed"]),
            "twist_topic": params["twist_topic"], "twist_is_stamped": self.twist_is_stamped,
            "odom_topic": params["odom_topic"], "imu_topic": params["imu_topic"],
            "camera_info_topic": params["camera_info_topic"],
            "save_fps_max": self.save_fps_max, "image_buffer_sec": self.image_buffer_sec, "process_hz": float(params["process_hz"]),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            import json; json.dump(meta, f, indent=2)

        self.get_logger().info("LiveTwistOnlyRecorder (NO crop/resize) initialized. Recording...")

    def destroy_node(self):
        try:
            if self.csv_file:
                self.csv_file.flush(); self.csv_file.close()
        except Exception:
            pass
        self.get_logger().info(f"Recorder summary: saved={self.saved}, seen={self.seen}, skipped_align={self.skipped_align}")
        return super().destroy_node()

    # --- Callbacks ---
    def on_image_common(self, img_bgr: np.ndarray, stamp_ns: int):
        self.seen += 1
        fname = f"frame_{stamp_ns}.png"
        self.images.append((stamp_ns, fname, img_bgr))
        cutoff = now_ns(self) - int(self.image_buffer_sec * 1e9)
        while self.images and self.images[0][0] < cutoff:
            self.images.popleft()

    def on_image(self, msg: Image):
        try: t_ns = ros_time_to_ns(msg.header.stamp) if msg.header else now_ns(self)
        except Exception: t_ns = now_ns(self)
        try:
            img_bgr = to_bgr_image_from_raw(msg)
            self.on_image_common(img_bgr, t_ns)
        except Exception as e:
            self.get_logger().warn(f"Failed to convert Image: {e}")

    def on_compressed_image(self, msg: CompressedImage):
        t_ns = get_msg_stamp_ns(self, msg)
        try:
            img_bgr = to_bgr_image_from_compressed(msg)
            self.on_image_common(img_bgr, t_ns)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode CompressedImage: {e}")

    def on_twist(self, msg: Twist):
        t_ns = now_ns(self)
        v = float(msg.linear.x); w = float(msg.angular.z)
        self.twist_ts.append(t_ns, {"v": v, "w": w})

    def on_twist_stamped(self, msg: 'TwistStamped'):
        t_ns = ros_time_to_ns(msg.header.stamp) if hasattr(msg, "header") else now_ns(self)
        tw = msg.twist if hasattr(msg, "twist") else msg
        v = float(tw.linear.x); w = float(tw.angular.z)
        self.twist_ts.append(t_ns, {"v": v, "w": w})

    def on_odom(self, msg: Odometry):
        t_ns = ros_time_to_ns(msg.header.stamp) if msg.header else now_ns(self)
        v = float(msg.twist.twist.linear.x); w = float(msg.twist.twist.angular.z)
        self.odom_ts.append(t_ns, {"v": v, "w": w})

    def on_imu(self, msg: Imu):
        t_ns = ros_time_to_ns(msg.header.stamp) if msg.header else now_ns(self)
        gz = float(msg.angular_velocity.z)
        ax = float(msg.linear_acceleration.x)
        self.imu_ts.append(t_ns, {"gyro_z": gz, "accel_x": ax})

    def on_camera_info(self, msg: CameraInfo):
        if self.cam_info_saved: return
        out = {
            "header": {"frame_id": msg.header.frame_id, "stamp_ns": ros_time_to_ns(msg.header.stamp) if msg.header else None},
            "height": int(msg.height), "width": int(msg.width),
            "distortion_model": msg.distortion_model, "D": list(msg.d),
            "K": list(msg.k), "R": list(msg.r), "P": list(msg.p),
            "binning_x": int(msg.binning_x), "binning_y": int(msg.binning_y),
        }
        with open(self.out_dir / "camera_info.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        self.cam_info_saved = True
        self.get_logger().info("Saved camera_info.yaml")

    # --- Processing ---
    def process_buffers(self):
        if not self.images or not self.twist_ts.t:
            return
        latest_twist = self.twist_ts.t[-1]
        processed = 0
        while self.images:
            t_img, fname, img_bgr = self.images[0]
            t_target = t_img + self.delay_ns
            if t_target > latest_twist:
                break  # wait for more control samples

            # Nearest matches
            twist_pair = self.twist_ts.nearest(t_target, self.tol_ns)
            if not twist_pair:
                self.images.popleft(); self.skipped_align += 1; continue
            twist_val, twist_src_t = twist_pair

            odom_pair = self.odom_ts.nearest(t_target, self.tol_ns) if self.odom_ts else None
            imu_pair  = self.imu_ts.nearest(t_target, self.tol_ns) if self.imu_ts else None

            # Rate cap
            if self.save_fps_max > 0.0 and self.last_save_ns > 0:
                min_dt = int(1e9 / self.save_fps_max)
                if t_img - self.last_save_ns < min_dt:
                    self.images.popleft(); self.skipped_align += 1; continue

            # Save image
            img_path = self.img_dir / fname
            ok = cv2.imwrite(str(img_path), img_bgr)
            if not ok:
                self.get_logger().warn(f"Failed to save image: {img_path}")
                self.images.popleft()
                continue

            # Accel estimate from Twist.v timeline (optional)
            a_val = self.twist_ts.accel_from_v(t_target, key="v")
            # Build row
            row = {
                "file": f"images/{fname}",
                "stamp_ns": int(t_img),
                "t_target_ns": int(t_target),
                "delay_ms_used": float(self.get_parameter('delay_ms').value),
                "img_w": int(img_bgr.shape[1]),
                "img_h": int(img_bgr.shape[0]),
                "twist_src_ns": int(twist_src_t),
                "twist_linear_x_mps": float(twist_val.get("v", 0.0)),
                "twist_angular_z_rps": float(twist_val.get("w", 0.0)),
                "accel_from_twist_mps2": (float(a_val) if a_val is not None else ""),
                "throttle_from_twist_norm": (max(0.0, min(1.0, a_val / float(self.get_parameter('a_max').value))) if (a_val is not None and float(self.get_parameter('a_max').value) > 0) else ""),
                "brake_from_twist_norm": (max(0.0, min(1.0, -a_val / float(self.get_parameter('b_max').value))) if (a_val is not None and float(self.get_parameter('b_max').value) > 0) else ""),
            }
            if odom_pair:
                odom_val, odom_src_t = odom_pair
                row["odom_src_ns"] = int(odom_src_t)
                row["odom_v_mps"]  = float(odom_val.get("v", 0.0))
                row["odom_w_rps"]  = float(odom_val.get("w", 0.0))
            else:
                row["odom_src_ns"] = ""; row["odom_v_mps"] = ""; row["odom_w_rps"] = ""
            if imu_pair:
                imu_val, imu_src_t = imu_pair
                row["imu_src_ns"]  = int(imu_src_t)
                row["imu_gyro_z_rps"] = float(imu_val.get("gyro_z", 0.0))
                row["imu_accel_x_mps2"] = float(imu_val.get("accel_x", 0.0))
            else:
                row["imu_src_ns"] = ""; row["imu_gyro_z_rps"] = ""; row["imu_accel_x_mps2"] = ""

            self.write_csv_row(row)
            self.saved += 1; self.last_save_ns = t_img; self.images.popleft(); processed += 1

        if processed > 0:
            self.get_logger().debug(f"Processed {processed} frames; total saved={self.saved}")

    def write_csv_row(self, row: dict):
        cols = [
            "file","stamp_ns","t_target_ns","delay_ms_used","img_w","img_h",
            "twist_src_ns","twist_linear_x_mps","twist_angular_z_rps",
            "accel_from_twist_mps2","throttle_from_twist_norm","brake_from_twist_norm",
            "odom_src_ns","odom_v_mps","odom_w_rps",
            "imu_src_ns","imu_gyro_z_rps","imu_accel_x_mps2"
        ]
        if not self.csv_header_written:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=cols)
            if self.csv_file.tell() == 0:
                self.csv_writer.writeheader()
            self.csv_header_written = True
        for k in cols:
            if k not in row: row[k] = ""
        self.csv_writer.writerow(row)
        self.csv_rows_since_flush += 1
        if self.csv_rows_since_flush >= self.csv_flush_every:
            self.csv_file.flush(); self.csv_rows_since_flush = 0

def main():
    rclpy.init()
    node = LiveTwistOnlyRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
