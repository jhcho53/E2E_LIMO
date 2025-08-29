# collect — 문서

## 주요 기능
- **실시간 로깅**: 카메라(`sensor_msgs/Image` 또는 `CompressedImage`) + 제어(`/cmd_vel: Twist/TwistStamped`)를 정렬하여 저장
- **레이턴시 보정**: 이미지 시각에 **미래(+delay_ms)** 의 제어를 매칭 (실차 지연 보정)
- **허용 오차 매칭**: 최근접 제어 샘플이 `±align_tol_ms` 안에 있을 때만 유효로 인정
- **무전처리 이미지 저장**: **크롭/리사이즈 없음** — 모델 학습 단계에서 원하는 전처리를 하세요
- **보조 신호(선택)**: `/odom`, `/imu`를 같은 기준 시각으로 맞춰 **검증/분석 컬럼**으로 저장
- **카메라 내부파라미터(선택)**: 최초 1회 `camera_info.yaml` 저장
- **저장 FPS 제한**: 저장 I/O 병목이나 용량 급증을 막기 위해 `save_fps_max`로 제한 가능

---

## 요구 사항
- ROS 2(rclpy 동작 환경), Python 3
- 패키지: `opencv-python-headless`, `numpy`, `pandas`, `pyyaml`
- (선택) Odometry/IMU 메시지 타입이 있는 워크스페이스

---

## 빠른 시작

```bash
pip install opencv-python-headless numpy pandas pyyaml

python3 live_e2e_recorder_twist_nocrop.py \
  --ros-args \
  -p image_topic:=/camera/front/image_raw \
  -p image_is_compressed:=false \
  -p twist_topic:=/cmd_vel \
  -p twist_is_stamped:=false \
  -p odom_topic:=/odom \                       # 선택
  -p imu_topic:=/imu/data \                    # 선택
  -p camera_info_topic:=/camera/front/camera_info \  # 선택
  -p out_dir:=/tmp/e2e_dataset_live \
  -p delay_ms:=120.0 \
  -p align_tol_ms:=80.0 \
  -p a_max:=1.5 -p b_max:=2.0 \
  -p save_fps_max:=15.0
