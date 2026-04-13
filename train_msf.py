"""
YOLO11-JDE 多尺度融合优化训练脚本

使用说明：
1. 本脚本提供三种多尺度融合方案
2. 建议先用较小分辨率测试，再逐步提升
3. 监控HOTA指标的提升

作者：基于YOLO11-JDE改进
日期：2026-03-09
"""
import sys
sys.path.append("/root/autodl-tmp/YOLO11-JDE-main")

import comet_ml
import os
import torch
from functools import partial
from tracker.evaluation.mot_callback import mot_eval
from ultralytics import YOLO
from datetime import datetime

# 初始化Comet日志
from ultralytics.utils import SETTINGS
SETTINGS['comet'] = True
comet_ml.init()

# ==================== 配置区 ====================

# 选择模型配置（三选一）
MODEL_CONFIGS = {
    'msf': 'yolo11s-jde-msf.yaml',      # 多尺度融合（推荐，效果最好）
    'aspp': 'yolo11s-jde-aspp.yaml',    # ASPP空洞金字塔（适合遮挡场景）
    'ca': 'yolo11s-jde-ca.yaml',        # 通道注意力（最轻量）
    'baseline': 'yolo11s-jde.yaml'      # 原始基线（对比用）
}

# 选择使用的模型
USE_MODEL = 'msf'  # 可选: 'msf', 'aspp', 'ca', 'baseline'

MODEL_YAML = MODEL_CONFIGS[USE_MODEL]
PRETRAIN = "ultralytics/weight/yolo11s.pt"
# DATA_YAML = "ultralytics/cfg/datasets/crowdhuman.yaml"
#MOT20的配置
DATA_YAML = "ultralytics/cfg/datasets/mot20.yaml"
TRACKER_YAML = "yolojdetracker.yaml"

# ==================== 训练参数 ====================

# 阶段1：基线测试（快速验证）
# epochs = 50
# batch = 8
# imgsz = 640

# 阶段2：提升分辨率（推荐）
epochs = 100
batch = 6  # 5090 32G应该能支持，如果OOM降到12
imgsz = 640

# 阶段3：完整训练（对标论文）
# epochs = 100
# batch = 24  # 根据显存调整
# imgsz = 1280

# ==================== 打印配置 ====================
print("=" * 60)
print(f"🚀 YOLO11-JDE 多尺度融合优化训练")
print("=" * 60)
print(f"模型配置: {USE_MODEL.upper()} - {MODEL_YAML}")
print(f"预训练权重: {PRETRAIN}")
print(f"数据集: {DATA_YAML}")
print(f"训练参数: epochs={epochs}, batch={batch}, imgsz={imgsz}")
print("=" * 60)

# ==================== 模型初始化 ====================
model = YOLO(MODEL_YAML, task='jde').load(PRETRAIN)

# 添加MOT评估回调
model.add_callback(
    "on_val_end",
    partial(mot_eval, period=epochs)
)

# ==================== 开始训练 ====================
model.train(
    project='reid_xps',
    name=f'CH-jde-{USE_MODEL}-b{batch}-e{epochs}-sz{imgsz}-' + datetime.now().strftime('%Y%m%d-%H%M%S'),
    
    data=DATA_YAML,
    epochs=epochs,
    batch=batch,
    device=[0,1],  # 使用单卡，多卡改为 [0,1,2,3]
    imgsz=imgsz,
    
    close_mosaic=0,     # JDE必须为0（自监督训练需要）
    patience=25,
    tracker=TRACKER_YAML,
    
    save=True,
    save_json=True,
    plots=True,
    verbose=True,
    cache=False,
    amp=True,           # 混合精度训练
    workers=8,          # 关闭多进程（避免Windows兼容问题）
)

print("\n" + "=" * 60)
print("✅ 训练完成！")
print("=" * 60)
print("\n📊 查看结果：")
print("1. 训练曲线：reid_xps/<实验名>/")
print("2. MOT指标：查看终端输出的HOTA/MOTA/IDF1")
print("3. Comet.ml：在线查看详细日志")
print("\n💡 优化建议：")
print("- 如果HOTA提升明显，可以继续增大batch和epochs")
print("- 如果显存不足，降低batch或imgsz")
print("- 对比不同模型配置的效果，选择最优方案")
