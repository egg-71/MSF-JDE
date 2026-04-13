# -*- coding: utf-8 -*-
"""
MOT20-04推理和可视化脚本
在测试集序列上运行模型推理，并可视化检测框和track ID

使用方法：
python visualize_mot20_inference.py

作者：基于YOLO11-JDE项目
日期：2026-03-22
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

# ========= 配置区 =========
# 模型权重路径
MODEL_PATH = "reid_xps/MOT20-baseline-xxx/weights/best.pt"  # 修改为实际路径

# MOT20数据集路径
MOT20_ROOT = Path(r"D:\Code\Data\MOT20")  # 或 YOLOTrack_MOT20
SEQUENCE = "MOT20-04"  # 要可视化的序列

# 输出路径
OUTPUT_DIR = Path(r"D:\Code\Data\MOT20_Visualization")

# 可视化参数
CONF_THRESH = 0.3  # 置信度阈值
SAVE_VIDEO = True  # 是否保存为视频
SAVE_IMAGES = True  # 是否保存单帧图片
MAX_FRAMES = None  # 最大帧数（None表示全部）
FPS = 25  # 视频帧率

# Tracker配置
TRACKER_CONFIG = "yolojdetracker.yaml"
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = OUTPUT_DIR / SEQUENCE / "frames"
if SAVE_IMAGES:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def get_colors(num_colors):
    """生成不同的颜色用于不同的track ID"""
    np.random.seed(42)
    colors = []
    for i in range(num_colors):
        # 使用HSV色彩空间生成鲜艳的颜色
        hue = int(180 * i / num_colors)
        color = cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), 
            cv2.COLOR_HSV2BGR
        )[0][0]
        colors.append(tuple(map(int, color)))
    return colors


# 预生成100种颜色
COLORS = get_colors(100)


def draw_track(img, box, track_id, conf, color):
    """
    在图片上绘制跟踪框和ID
    
    Args:
        img: 图片
        box: [x1, y1, x2, y2]
        track_id: 轨迹ID
        conf: 置信度
        color: 颜色
    """
    x1, y1, x2, y2 = map(int, box)
    
    # 绘制矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 准备文本
    label = f"ID:{track_id} {conf:.2f}"
    
    # 计算文本大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    
    # 绘制文本背景
    cv2.rectangle(
        img, 
        (x1, y1 - text_h - baseline - 5), 
        (x1 + text_w, y1), 
        color, 
        -1
    )
    
    # 绘制文本
    cv2.putText(
        img, 
        label, 
        (x1, y1 - baseline - 5), 
        font, 
        font_scale, 
        (255, 255, 255), 
        thickness
    )
    
    return img


def process_sequence(model, seq_path, output_dir):
    """
    处理一个序列
    
    Args:
        model: YOLO模型
        seq_path: 序列路径
        output_dir: 输出目录
    """
    print(f"\n处理序列: {seq_path.name}")
    
    # 获取图片目录
    img_dir = seq_path / "img1"
    if not img_dir.exists():
        print(f"[错误] 图片目录不存在: {img_dir}")
        return
    
    # 获取所有图片
    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_files:
        print(f"[错误] 没有找到图片")
        return
    
    # 限制帧数
    if MAX_FRAMES is not None:
        img_files = img_files[:MAX_FRAMES]
    
    print(f"总帧数: {len(img_files)}")
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(str(img_files[0]))
    h, w = first_frame.shape[:2]
    
    # 初始化视频写入器
    video_writer = None
    if SAVE_VIDEO:
        video_path = output_dir / f"{seq_path.name}_tracking.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_path), fourcc, FPS, (w, h)
        )
        print(f"视频输出: {video_path}")
    
    # 统计信息
    total_detections = 0
    unique_ids = set()
    
    # 处理每一帧
    for frame_idx, img_path in enumerate(tqdm(img_files, desc="推理中")):
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[警告] 无法读取: {img_path}")
            continue
        
        # 运行跟踪
        results = model.track(
            img, 
            persist=True,  # 保持track ID
            tracker=TRACKER_CONFIG,
            conf=CONF_THRESH,
            verbose=False
        )
        
        # 获取结果
        if results and len(results) > 0:
            result = results[0]
            
            # 检查是否有检测结果
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
                confs = result.boxes.conf.cpu().numpy()
                
                # 获取track IDs
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = np.arange(len(boxes))  # 如果没有ID，使用索引
                
                # 绘制每个检测框
                for box, track_id, conf in zip(boxes, track_ids, confs):
                    # 选择颜色
                    color = COLORS[track_id % len(COLORS)]
                    
                    # 绘制
                    img = draw_track(img, box, track_id, conf, color)
                    
                    # 统计
                    total_detections += 1
                    unique_ids.add(track_id)
        
        # 添加帧信息
        frame_info = f"Frame: {frame_idx+1}/{len(img_files)} | Objects: {len(boxes) if 'boxes' in locals() else 0}"
        cv2.putText(
            img, frame_info, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        
        # 保存单帧
        if SAVE_IMAGES:
            frame_path = FRAMES_DIR / f"{frame_idx+1:06d}.jpg"
            cv2.imwrite(str(frame_path), img)
        
        # 写入视频
        if video_writer is not None:
            video_writer.write(img)
    
    # 释放资源
    if video_writer is not None:
        video_writer.release()
    
    # 打印统计
    print(f"\n统计信息:")
    print(f"  总帧数: {len(img_files)}")
    print(f"  总检测数: {total_detections}")
    print(f"  唯一ID数: {len(unique_ids)}")
    print(f"  平均每帧: {total_detections/len(img_files):.1f} 个目标")
    
    if SAVE_VIDEO:
        print(f"\n✓ 视频已保存: {video_path}")
    if SAVE_IMAGES:
        print(f"✓ 单帧已保存: {FRAMES_DIR}")


def main():
    print("=" * 60)
    print("MOT20推理和可视化工具")
    print("=" * 60)
    
    # 检查模型路径
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"\n[错误] 模型权重不存在: {model_path}")
        print("\n请修改脚本中的 MODEL_PATH 变量")
        print("例如: reid_xps/MOT20-baseline-xxx/weights/best.pt")
        return
    
    print(f"\n模型权重: {model_path}")
    
    # 加载模型
    print("\n加载模型...")
    model = YOLO(model_path, task='jde')
    print("✓ 模型加载成功")
    
    # 检查序列路径
    seq_path = MOT20_ROOT / "test" / SEQUENCE
    if not seq_path.exists():
        # 尝试从YOLO格式数据集中找
        seq_path = MOT20_ROOT / "images" / "test"
        if not seq_path.exists():
            print(f"\n[错误] 序列不存在: {SEQUENCE}")
            print(f"请检查路径: {MOT20_ROOT}")
            return
    
    print(f"序列路径: {seq_path}")
    print(f"输出路径: {OUTPUT_DIR}")
    print(f"置信度阈值: {CONF_THRESH}")
    print(f"保存视频: {SAVE_VIDEO}")
    print(f"保存单帧: {SAVE_IMAGES}")
    
    # 处理序列
    process_sequence(model, seq_path, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
