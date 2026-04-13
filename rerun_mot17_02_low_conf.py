"""
使用更低的置信度阈值重新推理MOT17-02
"""
import sys
sys.path.append("/root/autodl-tmp/YOLO11-JDE-main")

import os
import cv2
import yaml
import time
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

from ultralytics import YOLO
from ultralytics.trackers import YOLOJDETracker

def dict_to_namespace(d):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})

# 配置 - 降低置信度阈值
MODEL_PATH = "reid_xps/CH-jde-msf-b8-e100-sz1280-20260313-190642/weights/best.pt"
OUTPUT_DIR = "reid_xps/CH-jde-msf-b8-e100-sz1280-20260313-190642/MOT17/val_half/data"
TRACKER_YAML = "yolojdetracker.yaml"
DEVICE = 0
IMGSZ = 1280
CONF_THRESHOLD = 0.05  # 从0.1降低到0.05
MAX_DET = 300

dataset_root = './tracker/evaluation/TrackEval/data/gt/mot_challenge/MOT17/val_half'
seq_name = 'MOT17-02-FRCNN'

print("=" * 80)
print(f"🔧 使用低置信度阈值({CONF_THRESHOLD})重新推理 {seq_name}")
print("=" * 80)

# 加载模型
print("加载模型...")
model = YOLO(MODEL_PATH, task='jde')

# 加载tracker配置
tracker_name = TRACKER_YAML.split('.')[0]
tracker_cfg = dict_to_namespace(yaml.safe_load(open(f"./ultralytics/cfg/trackers/{TRACKER_YAML}")))

# 获取图像列表
img_dir = os.path.join(dataset_root, seq_name, 'img1')
imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

print(f"序列: {seq_name}")
print(f"帧数: {len(imgs)}")
print(f"置信度阈值: {CONF_THRESHOLD}")

# 初始化tracker
tracker = YOLOJDETracker(args=tracker_cfg, frame_rate=30)

sequence_data_list = []
total_time = 0.0
total_detections = 0

for idx, img in enumerate(tqdm(imgs, desc="处理帧")):
    img_path = os.path.join(img_dir, img)
    img_file = cv2.imread(img_path)

    if img_file is None:
        continue

    # 预热模型（第一帧）
    if idx == 0:
        print("预热模型...")
        for _ in range(10):
            _ = model.predict(
                source=img_path,
                verbose=False,
                save=False,
                conf=CONF_THRESHOLD,
                imgsz=IMGSZ,
                max_det=MAX_DET,
                device=DEVICE,
                half=False,
                classes=[0],
            )[0]

    # 推理
    start_time = time.time()
    result = model.predict(
        source=img_path,
        verbose=False,
        save=False,
        conf=CONF_THRESHOLD,
        imgsz=IMGSZ,
        max_det=MAX_DET,
        device=DEVICE,
        half=False,
        classes=[0],
    )[0]

    # 统计检测数
    total_detections += len(result.boxes)

    # 处理检测结果
    det = result.boxes.cpu().numpy()

    # 更新tracker
    embeds = result.embeds.data.cpu().numpy()
    tracks = tracker.update(det, img_file, embeds)

    # 更新计数器
    frame_time = time.time() - start_time
    total_time += frame_time

    # 处理跟踪结果
    if len(tracks) == 0:
        continue

    frame_data = np.hstack([
        (np.ones_like(tracks[:, 0]) * (idx + 1)).reshape(-1, 1),
        tracks[:, 4].reshape(-1, 1),
        tracks[:, :4],
    ])
    sequence_data_list.append(frame_data)

# 保存结果
if len(sequence_data_list) > 0:
    sequence_data = np.vstack(sequence_data_list)

    # 转换bbox格式
    sequence_data[:, 4] -= sequence_data[:, 2]
    sequence_data[:, 5] -= sequence_data[:, 3]

    # 添加额外列
    constant_cols = np.ones((sequence_data.shape[0], 4)) * -1
    sequence_data = np.hstack([sequence_data, constant_cols])

    # 保存
    output_path = os.path.join(OUTPUT_DIR, f'{seq_name}.txt')

    # 备份
    if os.path.exists(output_path):
        backup_path = output_path + '.backup_conf01'
        os.rename(output_path, backup_path)
        print(f"已备份旧文件(conf=0.1): {backup_path}")

    with open(output_path, 'w') as file:
        np.savetxt(file, sequence_data, fmt='%.6f', delimiter=',')

    print(f"\n✅ 完成!")
    print(f"总检测数: {total_detections}")
    print(f"平均每帧检测数: {total_detections / len(imgs):.1f}")
    print(f"轨迹数: {len(sequence_data)}")
    print(f"平均每帧轨迹数: {len(sequence_data) / len(imgs):.1f}")
    print(f"FPS: {len(imgs) / total_time:.2f}")
    print(f"输出文件: {output_path}")
else:
    print("\n❌ 没有检测到任何目标！")

print("=" * 80)