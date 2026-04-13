# -*- coding: utf-8 -*-
"""
MOT20-04推理和可视化脚本（服务器版本）
在服务器上运行，生成可视化结果

使用方法：
python visualize_mot20_server.py

作者：基于YOLO11-JDE项目
日期：2026-03-22
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

# ========= 配置区（服务器路径）=========
# 模型权重路径（修改为您训练好的模型）
MODEL_PATH = "reid_xps/MOT20-baseline-b16-e100-sz1280-20260322-XXXXXX/weights/best.pt"

# MOT20数据集路径
MOT20_ROOT = Path("/root/autodl-tmp/YOLOTrack_MOT20")
SEQUENCE = "MOT20-04"  # 测试序列

# 输出路径
OUTPUT_DIR = Path("/root/autodl-tmp/MOT20_Visualization")

# 可视化参数
CONF_THRESH = 0.3
SAVE_VIDEO = True
SAVE_IMAGES = False  # 服务器上建议False（节省空间）
MAX_FRAMES = 300  # 限制帧数（测试用，None表示全部）
FPS = 25

TRACKER_CONFIG = "yolojdetracker.yaml"
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_colors(num_colors):
    """生成颜色"""
    np.random.seed(42)
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        color = cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), 
            cv2.COLOR_HSV2BGR
        )[0][0]
        colors.append(tuple(map(int, color)))
    return colors


COLORS = get_colors(200)  # 预生成200种颜色


def draw_track(img, box, track_id, conf, color):
    """绘制跟踪框"""
    x1, y1, x2, y2 = map(int, box)
    
    # 绘制矩形
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 文本
    label = f"ID:{track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    
    # 背景
    cv2.rectangle(
        img, 
        (x1, y1 - text_h - baseline - 5), 
        (x1 + text_w, y1), 
        color, 
        -1
    )
    
    # 文本
    cv2.putText(
        img, label, (x1, y1 - baseline - 5), 
        font, font_scale, (255, 255, 255), thickness
    )
    
    return img


def process_sequence(model, img_dir, output_dir, seq_name):
    """处理序列"""
    print(f"\n处理序列: {seq_name}")
    
    if not img_dir.exists():
        print(f"[错误] 图片目录不存在: {img_dir}")
        return
    
    # 获取图片列表
    img_files = sorted(img_dir.glob(f"{seq_name}_*.jpg")) + \
                sorted(img_dir.glob(f"{seq_name}_*.png"))
    
    if not img_files:
        print(f"[错误] 没有找到图片")
        return
    
    if MAX_FRAMES is not None:
        img_files = img_files[:MAX_FRAMES]
    
    print(f"总帧数: {len(img_files)}")
    
    # 读取第一帧
    first_frame = cv2.imread(str(img_files[0]))
    h, w = first_frame.shape[:2]
    
    # 视频写入器
    video_writer = None
    if SAVE_VIDEO:
        video_path = output_dir / f"{seq_name}_tracking.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_path), fourcc, FPS, (w, h)
        )
        print(f"视频输出: {video_path}")
    
    # 统计
    total_detections = 0
    unique_ids = set()
    frame_detections = []
    
    # 处理每帧
    for frame_idx, img_path in enumerate(tqdm(img_files, desc="推理中")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 跟踪
        results = model.track(
            img, 
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=CONF_THRESH,
            verbose=False
        )
        
        num_objects = 0
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = np.arange(len(boxes))
                
                num_objects = len(boxes)
                
                # 绘制
                for box, track_id, conf in zip(boxes, track_ids, confs):
                    color = COLORS[track_id % len(COLORS)]
                    img = draw_track(img, box, track_id, conf, color)
                    total_detections += 1
                    unique_ids.add(track_id)
        
        frame_detections.append(num_objects)
        
        # 帧信息
        frame_info = f"Frame: {frame_idx+1}/{len(img_files)} | Objects: {num_objects} | IDs: {len(unique_ids)}"
        cv2.putText(
            img, frame_info, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # 保存
        if SAVE_IMAGES:
            frames_dir = output_dir / seq_name / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frames_dir / f"{frame_idx+1:06d}.jpg"), img)
        
        if video_writer is not None:
            video_writer.write(img)
    
    if video_writer is not None:
        video_writer.release()
    
    # 统计
    print(f"\n统计信息:")
    print(f"  总帧数: {len(img_files)}")
    print(f"  总检测数: {total_detections}")
    print(f"  唯一ID数: {len(unique_ids)}")
    print(f"  平均每帧: {total_detections/len(img_files):.1f} 个目标")
    print(f"  最大每帧: {max(frame_detections)} 个目标")
    print(f"  最小每帧: {min(frame_detections)} 个目标")
    
    if SAVE_VIDEO:
        print(f"\n✓ 视频已保存: {video_path}")
        print(f"  可以下载到本地查看")


def main():
    print("=" * 60)
    print("MOT20推理和可视化工具（服务器版）")
    print("=" * 60)
    
    # 检查模型
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"\n[错误] 模型权重不存在: {model_path}")
        print("\n可用的模型:")
        reid_xps = Path("reid_xps")
        if reid_xps.exists():
            for exp_dir in sorted(reid_xps.iterdir()):
                if exp_dir.is_dir():
                    best_pt = exp_dir / "weights" / "best.pt"
                    if best_pt.exists():
                        print(f"  - {best_pt}")
        return
    
    print(f"\n模型权重: {model_path}")
    
    # 加载模型
    print("\n加载模型...")
    model = YOLO(model_path, task='jde')
    print("✓ 模型加载成功")
    
    # 检查数据集
    img_dir = MOT20_ROOT / "images" / "test"
    if not img_dir.exists():
        print(f"\n[错误] 测试集不存在: {img_dir}")
        return
    
    print(f"数据集路径: {MOT20_ROOT}")
    print(f"序列: {SEQUENCE}")
    print(f"输出路径: {OUTPUT_DIR}")
    print(f"置信度阈值: {CONF_THRESH}")
    print(f"最大帧数: {MAX_FRAMES if MAX_FRAMES else '全部'}")
    
    # 处理
    process_sequence(model, img_dir, OUTPUT_DIR, SEQUENCE)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n下载可视化结果:")
    print(f"scp username@server:{OUTPUT_DIR / f'{SEQUENCE}_tracking.mp4'} .")


if __name__ == "__main__":
    main()
