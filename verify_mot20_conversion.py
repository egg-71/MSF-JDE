# -*- coding: utf-8 -*-
"""
验证MOT20转YOLO格式的转换结果
检查图片和标注是否正确对应，标注格式是否正确

使用方法：
python verify_mot20_conversion.py

作者：基于YOLO11-JDE项目
日期：2026-03-22
"""

import os
import cv2
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ========= 配置区 =========
YOLO_ROOT = Path(r"D:\Code\Data\YOLOTrack_MOT20")
NUM_SAMPLES = 5  # 每个split随机抽取的样本数
# =========================


def check_label_format(label_path: Path):
    """检查标注文件格式"""
    errors = []
    
    if not label_path.exists():
        return ["标注文件不存在"]
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    if not lines:
        return ["标注文件为空"]
    
    for i, line in enumerate(lines, 1):
        parts = line.strip().split()
        
        # YOLO JDE格式：class cx cy w h track_id
        if len(parts) != 6:
            errors.append(f"第{i}行: 应该有6列，实际有{len(parts)}列")
            continue
        
        try:
            cls, cx, cy, w, h, tid = parts
            cls = int(cls)
            cx, cy, w, h = map(float, [cx, cy, w, h])
            tid = int(tid)
            
            # 检查数值范围
            if not (0.0 <= cx <= 1.0):
                errors.append(f"第{i}行: cx={cx} 超出[0,1]范围")
            if not (0.0 <= cy <= 1.0):
                errors.append(f"第{i}行: cy={cy} 超出[0,1]范围")
            if not (0.0 < w <= 1.0):
                errors.append(f"第{i}行: w={w} 超出(0,1]范围")
            if not (0.0 < h <= 1.0):
                errors.append(f"第{i}行: h={h} 超出(0,1]范围")
            if cls != 0:
                errors.append(f"第{i}行: class={cls}，应该为0")
            if tid < 0:
                errors.append(f"第{i}行: track_id={tid} 为负数")
                
        except ValueError as e:
            errors.append(f"第{i}行: 数值解析错误 - {e}")
    
    return errors


def visualize_sample(img_path: Path, label_path: Path, save_path: Path):
    """可视化一个样本"""
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [错误] 无法读取图片: {img_path}")
        return False
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    # 读取标注
    boxes = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 6:
                    cls, cx, cy, w, h, tid = parts
                    boxes.append({
                        'cx': float(cx),
                        'cy': float(cy),
                        'w': float(w),
                        'h': float(h),
                        'tid': int(tid)
                    })
    
    # 绘制
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # 绘制每个框
    for box in boxes:
        # 转换回像素坐标
        cx_px = box['cx'] * W
        cy_px = box['cy'] * H
        w_px = box['w'] * W
        h_px = box['h'] * H
        
        # 计算左上角坐标
        x1 = cx_px - w_px / 2
        y1 = cy_px - h_px / 2
        
        # 绘制矩形
        rect = patches.Rectangle(
            (x1, y1), w_px, h_px,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加track ID标签
        ax.text(
            x1, y1 - 5,
            f"ID:{box['tid']}",
            color='red', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    ax.axis('off')
    ax.set_title(f"{img_path.name}\n{len(boxes)} objects", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return True


def verify_split(split: str):
    """验证一个split（train/val/test）"""
    print(f"\n{'='*60}")
    print(f"验证 {split.upper()} 集")
    print('='*60)
    
    img_dir = YOLO_ROOT / "images" / split
    lbl_dir = YOLO_ROOT / "labels" / split
    
    if not img_dir.exists():
        print(f"[跳过] 图片目录不存在: {img_dir}")
        return
    
    # 统计信息
    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    n_images = len(img_files)
    
    if n_images == 0:
        print(f"[警告] 没有找到图片")
        return
    
    print(f"\n图片数量: {n_images}")
    
    # 检查标注文件
    if split != "test":  # 测试集没有标注
        if not lbl_dir.exists():
            print(f"[错误] 标注目录不存在: {lbl_dir}")
            return
        
        lbl_files = list(lbl_dir.glob("*.txt"))
        n_labels = len(lbl_files)
        print(f"标注数量: {n_labels}")
        
        if n_images != n_labels:
            print(f"[警告] 图片和标注数量不匹配！")
        
        # 检查对应关系
        print("\n检查图片和标注对应关系...")
        missing_labels = []
        for img_file in img_files:
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                missing_labels.append(img_file.name)
        
        if missing_labels:
            print(f"[警告] {len(missing_labels)} 张图片缺少标注文件")
            if len(missing_labels) <= 5:
                for name in missing_labels:
                    print(f"  - {name}")
        else:
            print("✓ 所有图片都有对应的标注文件")
        
        # 检查标注格式
        print("\n检查标注格式...")
        error_files = []
        total_boxes = 0
        
        for lbl_file in lbl_files[:100]:  # 只检查前100个
            errors = check_label_format(lbl_file)
            if errors:
                error_files.append((lbl_file.name, errors))
            else:
                with open(lbl_file, "r") as f:
                    total_boxes += len(f.readlines())
        
        if error_files:
            print(f"[警告] {len(error_files)} 个标注文件有格式错误")
            for name, errors in error_files[:3]:  # 只显示前3个
                print(f"\n  {name}:")
                for err in errors[:5]:  # 每个文件只显示前5个错误
                    print(f"    - {err}")
        else:
            print("✓ 标注格式检查通过")
        
        print(f"\n总标注框数: {total_boxes}")
        if n_labels > 0:
            print(f"平均每张图: {total_boxes/n_labels:.1f} 个框")
    
    # 可视化样本
    print(f"\n生成可视化样本...")
    vis_dir = YOLO_ROOT / "visualization" / split
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 随机抽取样本
    sample_imgs = random.sample(img_files, min(NUM_SAMPLES, len(img_files)))
    
    for img_file in sample_imgs:
        lbl_file = lbl_dir / f"{img_file.stem}.txt" if split != "test" else None
        save_path = vis_dir / f"{img_file.stem}_vis.jpg"
        
        success = visualize_sample(img_file, lbl_file, save_path)
        if success:
            print(f"  ✓ {img_file.name}")
    
    print(f"\n可视化结果保存在: {vis_dir}")


def check_yaml_config():
    """检查YAML配置文件"""
    print(f"\n{'='*60}")
    print("检查配置文件")
    print('='*60)
    
    yaml_path = YOLO_ROOT / "mot20.yaml"
    if not yaml_path.exists():
        print(f"[警告] 配置文件不存在: {yaml_path}")
        return
    
    print(f"\n配置文件: {yaml_path}")
    print("\n内容:")
    with open(yaml_path, "r", encoding="utf-8") as f:
        content = f.read()
        print(content)
    
    print("✓ 配置文件存在")


def main():
    print("=" * 60)
    print("MOT20转YOLO格式 - 验证工具")
    print("=" * 60)
    
    if not YOLO_ROOT.exists():
        print(f"\n[错误] YOLO数据集路径不存在: {YOLO_ROOT}")
        print("请先运行 convert_mot20_to_yolo.py 进行转换")
        return
    
    print(f"\n数据集路径: {YOLO_ROOT}")
    
    # 检查配置文件
    check_yaml_config()
    
    # 验证各个split
    for split in ["train", "val", "test"]:
        verify_split(split)
    
    # 总结
    print(f"\n{'='*60}")
    print("验证完成！")
    print('='*60)
    print("\n请检查:")
    print("1. 图片和标注数量是否匹配")
    print("2. 标注格式是否正确")
    print("3. 可视化结果是否正常")
    print(f"\n可视化结果位置: {YOLO_ROOT / 'visualization'}")
    print("\n如果一切正常，可以将数据集上传到服务器进行训练。")
    print("=" * 60)


if __name__ == "__main__":
    main()
