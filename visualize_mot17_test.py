"""
MOT17测试集可视化脚本
"""
import os
import sys
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace

sys.path.append("/root/autodl-tmp/YOLO11-JDE-main")

from ultralytics import YOLO
from ultralytics.trackers import YOLOJDETracker


def dict_to_namespace(d):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def get_colors(num_colors):
    np.random.seed(42)
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors


COLORS = get_colors(200)


def draw_track(img, box, track_id, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - baseline - 5), font, font_scale, (255, 255, 255), thickness)

    return img


def visualize_mot17_test(model_path, test_root, output_dir, tracker_yaml="yolojdetracker.yaml",
                         save_video=True, save_images=False, max_frames=None):
    """
    可视化MOT17测试集

    Args:
        test_root: 测试集根目录，例如 /root/autodl-tmp/YOLOTrack/images/test
    """
    print("=" * 60)
    print("🎬 MOT17测试集可视化")
    print("=" * 60)

    test_root = Path(test_root)

    # 获取所有测试序列文件夹
    sequences = sorted([d for d in test_root.iterdir() if d.is_dir()])

    print(f"找到 {len(sequences)} 个测试序列:")
    for seq in sequences:
        img_dir = seq / "img1"
        if img_dir.exists():
            num_imgs = len(list(img_dir.glob("*.jpg")))
            print(f"  - {seq.name}: {num_imgs} 帧")

    # 创建输出目录
    vis_dir = Path(output_dir) / "MOT17_test_visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n🔧 加载模型: {model_path}")
    model = YOLO(model_path, task='jde')

    # 加载tracker配置
    tracker_name = tracker_yaml.split('.')[0]
    tracker_cfg = dict_to_namespace(yaml.safe_load(open(f"./ultralytics/cfg/trackers/{tracker_name}.yaml")))

    # 处理每个序列
    for seq_path in sequences:
        seq_name = seq_path.name
        img_dir = seq_path / "img1"

        if not img_dir.exists():
            print(f"\n跳过 {seq_name}: 没有img1文件夹")
            continue

        print(f"\n处理序列: {seq_name}")

        # 获取图像列表
        imgs = sorted(img_dir.glob("*.jpg"))

        if max_frames:
            imgs = imgs[:max_frames]

        if len(imgs) == 0:
            print(f"  跳过: 没有图片")
            continue

        # 读取第一帧获取尺寸
        first_img = cv2.imread(str(imgs[0]))
        h, w = first_img.shape[:2]

        # 初始化视频写入器
        video_writer = None
        if save_video:
            video_path = vis_dir / f"{seq_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 25, (w, h))

        # 初始化tracker
        tracker = YOLOJDETracker(args=tracker_cfg, frame_rate=30)

        # 创建图片输出目录
        if save_images:
            img_out_dir = vis_dir / seq_name
            img_out_dir.mkdir(exist_ok=True)

        # 处理每一帧
        for idx, img_path in enumerate(tqdm(imgs, desc=f"  {seq_name}")):
            img = cv2.imread(str(img_path))

            # 推理
            result = model.predict(
                source=str(img_path),
                verbose=False,
                save=False,
                conf=0.1,
                imgsz=1280,
                max_det=300,
                device=0,
                half=False,
                classes=[0],
            )[0]

            # 获取检测结果
            det = result.boxes.cpu().numpy()
            embeds = result.embeds.data.cpu().numpy()

            # 更新tracker
            tracks = tracker.update(det, img, embeds)

            # 绘制跟踪结果
            if len(tracks) > 0:
                for track in tracks:
                    box = track[:4]
                    track_id = int(track[4])
                    color = COLORS[track_id % len(COLORS)]
                    img = draw_track(img, box, track_id, color)

            # 添加帧信息
            frame_info = f"Frame: {idx+1}/{len(imgs)} | Objects: {len(tracks)}"
            cv2.putText(img, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 保存
            if save_images:
                cv2.imwrite(str(img_out_dir / img_path.name), img)

            if video_writer:
                video_writer.write(img)

        if video_writer:
            video_writer.release()
            print(f"  ✅ 视频已保存: {video_path}")

    print("\n" + "=" * 60)
    print(f"✅ 可视化完成！输出目录: {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # 配置
    MODEL_PATH = "reid_xps/CH-jde-msf-b8-e100-sz1280-20260313-190642/weights/best.pt"
    TEST_ROOT = "/root/autodl-tmp/YOLOTrack/images/test"
    OUTPUT_DIR = "reid_xps/CH-jde-msf-b8-e100-sz1280-20260313-190642"

    # 运行可视化
    visualize_mot17_test(
        model_path=MODEL_PATH,
        test_root=TEST_ROOT,
        output_dir=OUTPUT_DIR,
        save_video=True,
        save_images=False,
        max_frames=None,
    )