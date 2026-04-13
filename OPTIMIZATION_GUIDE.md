# YOLO11-JDE 优化指南

## 📊 当前实验数据分析

### 您的实验（5090 32G）
- **参数**: epochs=50, batch=8, imgsz=640
- **结果**: HOTA=56.418, MOTA=66.695, IDF1=71.622, FPS=55.637

### 原论文（3090×8训练，L40推理）
- **参数**: epochs=100, batch=64, imgsz=1280
- **结果**: HOTA=60.06, MOTA=58.25, IDF1=71.84, FPS=-

### 关键发现
✅ **优势**：您的MOTA更高（66.695 vs 58.25），说明检测质量好  
⚠️ **待提升**：HOTA较低（56.418 vs 60.06），说明时序关联一致性需要提升  
✅ **持平**：IDF1相近（71.622 vs 71.84），ReID特征质量相当

---

## 🎯 优化方向（按优先级）

### 优先级1：提升训练参数 ⭐⭐⭐⭐⭐
**最直接有效的方法**

#### 建议配置
```python
# 阶段1：提升分辨率（推荐先做）
epochs = 100
batch = 16  # 5090 32G应该能支持
imgsz = 1280

# 阶段2：如果显存允许，继续提升
batch = 24  # 或更大
```

#### 预期效果
- HOTA: +2~3
- MOTA: 保持或略降
- IDF1: +1~2
- FPS: 降低（但推理时可用TensorRT优化）

#### 原因分析
1. **分辨率640→1280**：小目标检测能力提升，减少ID切换
2. **Batch 8→16+**：更稳定的梯度，更好的ReID特征学习
3. **Epochs 50→100**：充分收敛，特别是ReID分支

---

### 优先级2：多尺度特征融合 ⭐⭐⭐⭐
**适合您的情况，推荐尝试**

#### 为什么适合？
1. ✅ 您的HOTA偏低 → 不同尺度目标关联有提升空间
2. ✅ 现有架构已有P3/P4/P5 → 融合方式可改进
3. ✅ MOT场景特点 → 行人尺度变化大（远近、遮挡）

#### 三种方案

##### 方案A：自适应多尺度融合（推荐）
```bash
python train_msf.py
# 修改 USE_MODEL = 'msf'
```

**特点**：
- 跨尺度ReID特征融合
- 自适应学习融合权重
- 对不同尺度目标更鲁棒

**预期提升**：
- HOTA: +1.5~2.5
- 参数增加: ~5%
- 速度影响: -5~10%

##### 方案B：ASPP空洞金字塔
```bash
python train_msf.py
# 修改 USE_MODEL = 'aspp'
```

**特点**：
- 多尺度感受野
- 捕获不同尺度上下文
- 对遮挡更鲁棒

**预期提升**：
- HOTA: +1~2
- 遮挡场景IDF1: +2~3
- 参数增加: ~8%

##### 方案C：通道注意力（轻量级）
```bash
python train_msf.py
# 修改 USE_MODEL = 'ca'
```

**特点**：
- 最轻量（参数+2%）
- 自动学习重要通道
- 速度影响最小

**预期提升**：
- HOTA: +0.5~1
- 速度影响: <3%

---

### 优先级3：数据增强优化 ⭐⭐⭐

#### 建议调整
```python
# 在train_msf.py中添加
model.train(
    # ... 其他参数
    hsv_h=0.015,      # 色调增强
    hsv_s=0.7,        # 饱和度增强
    hsv_v=0.4,        # 亮度增强
    degrees=10.0,     # 旋转角度
    translate=0.1,    # 平移
    scale=0.5,        # 缩放
    shear=0.0,        # 剪切
    perspective=0.0,  # 透视
    flipud=0.0,       # 上下翻转
    fliplr=0.5,       # 左右翻转（行人对称）
    mixup=0.0,        # 不用mixup（会破坏ReID）
)
```

---

### 优先级4：Tracker超参数调优 ⭐⭐⭐

#### 使用自动调优
```bash
cd tracker/finetune
python evolve.py
```

#### 手动调整关键参数
编辑 `ultralytics/cfg/trackers/yolojdetracker.yaml`：

```yaml
# 关键参数说明
track_high_thresh: 0.6    # 高置信度阈值（降低可增加召回）
track_low_thresh: 0.1     # 低置信度阈值
new_track_thresh: 0.7     # 新轨迹阈值（提高可减少误检）

first_match_thresh: 0.8   # 第一次匹配阈值
second_match_thresh: 0.5  # 第二次匹配阈值

appearance_thresh: 0.25   # ReID相似度阈值（关键！）
proximity_thresh: 0.5     # 位置距离阈值
appearance_weight: 0.75   # ReID权重（0.5-0.9）

track_buffer: 30          # 轨迹缓存帧数（30fps基准）
```

**调优建议**：
- HOTA低 → 降低`track_high_thresh`，提高`appearance_weight`
- ID切换多 → 降低`appearance_thresh`，提高`track_buffer`
- 误检多 → 提高`new_track_thresh`

---

## 🚀 推荐实验流程

### 第1步：基线对比（1-2天）
```bash
# 1. 提升分辨率基线
python train.py
# 修改: epochs=100, batch=16, imgsz=1280

# 2. 多尺度融合版本
python train_msf.py
# USE_MODEL = 'msf'
```

**目标**：验证多尺度融合是否有效

---

### 第2步：选择最优方案（1天）
对比三种融合方案（msf/aspp/ca），选择HOTA提升最大的

---

### 第3步：精细调优（2-3天）
1. 调整tracker超参数
2. 尝试不同的数据增强
3. 如果显存允许，继续提升batch

---

### 第4步：完整训练（3-5天）
```python
epochs = 100
batch = 24  # 或更大
imgsz = 1280
```

---

## 📈 预期最终结果

基于您当前的基础（MOTA已经很高），预期优化后：

| 指标 | 当前 | 优化后（保守） | 优化后（理想） |
|------|------|----------------|----------------|
| HOTA | 56.4 | 58.5~59.5 | 60.0~61.0 |
| MOTA | 66.7 | 65.0~67.0 | 66.0~68.0 |
| IDF1 | 71.6 | 72.5~73.5 | 73.5~75.0 |
| FPS  | 55.6 | 45~50 | 40~45 |

---

## 💡 其他优化方向（可选）

### 1. 知识蒸馏
用更大的模型（yolo11l/x）蒸馏到yolo11s

### 2. 后处理优化
- 轨迹平滑（卡尔曼滤波参数调优）
- 线性插值填补短暂遮挡

### 3. 集成方法
- 多尺度测试（TTA）
- 多模型融合

### 4. 特定场景优化
- 如果是固定摄像头：加入背景建模
- 如果是移动摄像头：加强GMC（全局运动补偿）

---

## ❓ 常见问题

### Q1: 显存不足怎么办？
```python
# 方案1：降低batch
batch = 8
imgsz = 1280

# 方案2：梯度累积
batch = 8
accumulate = 2  # 等效batch=16

# 方案3：降低分辨率
batch = 16
imgsz = 960  # 介于640和1280之间
```

### Q2: 训练很慢怎么办？
```python
# 1. 启用缓存（如果内存够）
cache = 'ram'  # 或 'disk'

# 2. 增加workers（Linux）
workers = 8

# 3. 使用更小的验证集
val_split = 0.1  # 只用10%验证
```

### Q3: 如何判断哪个方案最好？
**主要看HOTA**（综合指标），其次看IDF1（ReID质量）

---

## 📞 技术支持

如有问题，请查看：
1. 原论文：[arXiv](https://arxiv.org/abs/2501.13710v1)
2. 代码仓库：检查issues
3. MOT Challenge：对比其他方法

---

## 📝 实验记录模板

建议记录每次实验：

```markdown
## 实验X：多尺度融合MSF

**日期**：2026-03-XX
**配置**：
- 模型：yolo11s-jde-msf
- 参数：epochs=100, batch=16, imgsz=1280
- 其他：无

**结果**：
- HOTA: XX.XX
- MOTA: XX.XX
- IDF1: XX.XX
- FPS: XX.XX

**结论**：
- HOTA提升了X.X
- 下一步：XXX
```

---

## 🎓 总结

**最推荐的优化路径**：
1. ✅ 提升分辨率到1280（必做）
2. ✅ 尝试多尺度融合MSF（推荐）
3. ✅ 调优tracker参数（重要）
4. ⭐ 增大batch和epochs（如果显存允许）

**预期时间**：1-2周完成完整实验

**预期提升**：HOTA 56.4 → 59~61

祝实验顺利！🚀
