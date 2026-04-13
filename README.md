# MSF-JDE: Multi-Scale Feature Fusion for Real-Time Multi-Object Tracking

We propose **MSF-JDE**, an enhanced Joint Detection and Embedding (JDE) framework built upon YOLO11-JDE for real-time multi-object tracking (MOT).
Our method introduces a **Multi-Scale Feature Fusion (MSF)** strategy into the Re-Identification (Re-ID) branch, enabling robust identity representation under challenging scenarios such as fast motion, occlusion, and scale variation.

By aggregating features from multiple pyramid levels (P3, P4, P5), MSF-JDE improves appearance consistency across frames and significantly reduces identity switches. The fusion weights are adaptively learned via a softmax-based normalization mechanism.

Extensive experiments on MOT17 demonstrate that MSF-JDE achieves substantial improvements in tracking accuracy while maintaining real-time performance.

---

##  Key Contributions

* **Multi-Scale Re-ID Feature Fusion (MSF)**

  * Introduces cross-scale fusion (P3/P4/P5) in the Re-ID branch
  * Enhances robustness to scale variation and occlusion

* **Adaptive Feature Weighting**

  * Learns scale importance via softmax normalization
  * Dynamically balances fine-grained and semantic features

* **Improved Identity Consistency**

  * Reduces ID switches in fast-motion scenarios (e.g., MOT17-02)
  * Enhances IDF1 and HOTA metrics

* **Real-Time Performance**

  * Maintains high FPS with minimal computational overhead

---

##  Experimental Results

### MOT17 Validation (val_half)

| Metric | Baseline (YOLO11-JDE) | MSF-JDE  |
| ------ | --------------------- | -------- |
| HOTA   | 54.1                  | **57.3** |
| MOTA   | 63.3                  | **68.4** |
| IDF1   | 68.8                  | **73.2** |
| FPS    | 56.7                  | 51.2     |

 **Performance Gain**:

* +6.0% HOTA
* +8.0% MOTA
* +6.4% IDF1

---

##  Method Overview

MSF-JDE enhances the Re-ID branch of YOLO11-JDE by introducing multi-scale feature fusion:

* Extract features from:

  * P3 (fine-grained details)
  * P4 (mid-level semantics)
  * P5 (high-level semantics)

* Align feature dimensions via convolution layers

* Apply adaptive weighting:

  ```
  w_i = Softmax(f_i)
  ```

* Fuse features:

  ```
  F = Σ w_i * F_i
  ```

This design improves robustness against:

* Occlusion
* Scale variation
* Fast motion

---

##  Dataset

* **CrowdHuman** (training)
* **MOT17** (validation: val_half)

Dataset should be converted to YOLO format.

---

##  Training

```bash
python train.py \
  --data crowdhuman.yaml \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8
```

---

##  Tracking

```bash
python track.py \
  --source MOT17 \
  --tracker yolojdetracker.yaml
```

---

## 📈 Key Insight

Baseline JDE struggles in sequences like **MOT17-02**, where:

* Rapid motion
* Frequent occlusion
* Identity switches

MSF-JDE solves this by:

 **leveraging multi-scale temporal-consistent features**

---

##  Future Work

* Temporal modeling (Transformer / LSTM)
* Motion-aware embedding refinement
* Integration with diffusion-based tracking

---

##  Acknowledgements

Based on:

* YOLO11-JDE
* Ultralytics YOLO

---

##  Citation

If you find this work useful, please cite:

```bibtex
@article{msfjde2026,
  title={MSF-JDE: Multi-Scale Feature Fusion for Real-Time Multi-Object Tracking},
  author={Your Name},
  year={2026}
}
```
