"""
独立MOT评估脚本
用于对已训练的模型进行MOT17评估
"""
import os
import sys
import cv2
import yaml
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace

# 添加项目路径
sys.path.append("/root/autodl-tmp/YOLO11-JDE-main")

from ultralytics import YOLO
from tracker.evaluation.evaluate import trackeval_evaluation
from ultralytics.trackers import YOLOJDETracker


def dict_to_namespace(d):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def run_mot_evaluation(model_path, output_dir, tracker_yaml="yolojdetracker.yaml"):
    """
    对指定模型运行MOT17评估

    参数:
        model_path: 模型权重路径 (例如: best.pt 或 last.pt)
        output_dir: 输出目录 (例如: reid_xps/CH-jde-msf-b8-e100-sz1280-20260313-190642)
        tracker_yaml: tracker配置文件
    """
    print("=" * 60)
    print("🚀 开始MOT17评估")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"输出目录: {output_dir}")
    print(f"Tracker: {tracker_yaml}")
    print("=" * 60)

    # 数据集配置
    dataset_name = 'MOT17/val_half'
    seqmap_file = './tracker/evaluation/TrackEval/data/gt/mot_challenge/seqmaps/MOT17-val_half.txt'
    dataset_root = os.path.join('./tracker/evaluation/TrackEval/data/gt/mot_challenge/', dataset_name)

    # 读取序列列表
    with open(seqmap_file, 'r') as f:
        seq_names = [line.strip() for line in f.readlines() if line.strip() and line.strip() != 'name']

    print(f"\n📋 将评估 {len(seq_names)} 个序列:")
    for seq in seq_names:
        print(f"  - {seq}")

    # 创建输出文件夹
    output_folder = os.path.join(output_dir, dataset_name, 'data')
    os.makedirs(output_folder, exist_ok=True)

    # 加载模型
    print(f"\n🔧 加载模型...")
    model = YOLO(model_path, task='jde')

    # 加载tracker配置
    tracker_name = tracker_yaml.split('.')[0]
    tracker_cfg = dict_to_namespace(yaml.safe_load(open(f"./ultralytics/cfg/trackers/{tracker_name}.yaml")))

    # 初始化计数器
    total_frames = 0
    total_time = 0.0

    # 遍历每个序列
    for seq_name in tqdm(seq_names, desc="🎬 处理序列"):
        # 获取图像列表
        imgs = sorted(os.listdir(os.path.join(dataset_root, seq_name, 'img1')))

        # 为每个序列重新初始化tracker
        tracker = YOLOJDETracker(args=tracker_cfg, frame_rate=30)

        sequence_data_list = []

        for idx, img in enumerate(tqdm(imgs, desc=f"  {seq_name}", leave=False)):
            if img.endswith('.jpg') or img.endswith('.png'):
                # 图像路径
                img_path = os.path.join(dataset_root, seq_name, 'img1', img)
                img_file = cv2.imread(img_path)

                # 预热模型(第一帧)
                if idx == 0:
                    for _ in range(10):
                        _ = model.predict(
                            source=img_path,
                            verbose=False,
                            save=False,
                            conf=0.1,
                            imgsz=1280,
                            max_det=300,
                            device=0,  # 使用单卡推理
                            half=False,
                            classes=[0],
                        )[0]

                # 推理
                start_time = time.time()
                result = model.predict(
                    source=img_path,
                    verbose=False,
                    save=False,
                    conf=0.1,
                    imgsz=1280,
                    max_det=300,
                    device=0,
                    half=False,
                    classes=[0],
                )[0]

                # 处理检测结果
                det = result.boxes.cpu().numpy()

                # 更新tracker
                embeds = result.embeds.data.cpu().numpy()
                tracks = tracker.update(det, img_file, embeds)

                # 更新计数器
                frame_time = time.time() - start_time
                total_time += frame_time
                total_frames += 1

                # 处理跟踪结果
                if len(tracks) == 0:
                    continue

                frame_data = np.hstack([
                    (np.ones_like(tracks[:, 0]) * (idx + 1)).reshape(-1, 1),  # Frame number
                    tracks[:, 4].reshape(-1, 1),  # Track ID
                    tracks[:, :4],  # Bbox XYXY
                ])
                sequence_data_list.append(frame_data)

        # 保存该序列的结果
        if len(sequence_data_list) > 0:
            sequence_data = np.vstack(sequence_data_list)

            # 转换bbox格式: TLBR -> TLWH
            sequence_data[:, 4] -= sequence_data[:, 2]
            sequence_data[:, 5] -= sequence_data[:, 3]

            # 添加置信度、类别、可见性等列
            constant_cols = np.ones((sequence_data.shape[0], 4)) * -1
            sequence_data = np.hstack([sequence_data, constant_cols])

            # 保存到文件
            txt_path = output_folder + f'/{seq_name}.txt'
            with open(txt_path, 'w') as file:
                np.savetxt(file, sequence_data, fmt='%.6f', delimiter=',')

            print(f"  ✅ {seq_name}: 保存 {len(sequence_data)} 条轨迹")

    # 打印性能统计
    print("\n" + "=" * 60)
    print("⏱️  性能统计")
    print("=" * 60)
    print(f"总帧数: {total_frames}")
    print(f"总时间: {total_time:.3f}s")
    print(f"平均FPS: {total_frames / total_time:.3f}")

    # 运行TrackEval评估
    print("\n" + "=" * 60)
    print("📊 运行TrackEval评估...")
    print("=" * 60)

    config = {
        'GT_FOLDER': dataset_root,
        'TRACKERS_FOLDER': '/'.join(output_folder.split('/')[:-1]),
        'TRACKERS_TO_EVAL': [''],
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': 4,
        'SKIP_SPLIT_FOL': True,
        'SEQMAP_FILE': seqmap_file,
        'PRINT_CONFIG': False,
        'PRINT_RESULTS': True,
    }

    trackeval_evaluation(config)

    # 读取并显示最终结果
    summary_path = '/'.join(output_folder.split('/')[:-1]) + '/pedestrian_summary.txt'
    summary_df = pd.read_csv(summary_path, sep=' ')
    hota, mota, idf1 = summary_df.loc[0, ['HOTA', 'MOTA', 'IDF1']]

    print("\n" + "=" * 60)
    print("🎯 最终评估结果")
    print("=" * 60)
    print(f"HOTA: {hota:.3f}")
    print(f"MOTA: {mota:.3f}")
    print(f"IDF1: {idf1:.3f}")
    print("=" * 60)
    print("✅ 评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "reid_xps/CH-jde-ca-b12-e30-sz1280-20260323-185746/weights/best.pt"
    OUTPUT_DIR = "reid_xps/CH-jde-ca-b12-e30-sz1280-20260323-185746"
    TRACKER_YAML = "yolojdetracker.yaml"

    # 运行评估
    run_mot_evaluation(MODEL_PATH, OUTPUT_DIR, TRACKER_YAML)