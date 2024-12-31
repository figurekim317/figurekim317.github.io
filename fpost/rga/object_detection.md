---
layout: fpost
title: "Object Detection Model Optimization for Enhanced Performance and Efficient Inference on Nvidia Orin"
permalink: /fpost/rga/object_detection/
author: Dohyeong Kim
tags:   
  - Real-time Object Detection
  - Model Compression
  - Optimization
  - TensorRT
  - Jetson Deployment
  - Nvidia Orin
---

## Goal
- Develop a **lightweight object detection model** capable of **real-time inference** on constrained devices like Nvidia Orin.
- Optimize baseline models for **enhanced accuracy**, **reduced latency**, and **deployment efficiency**.

---

### 1. Object Detection: Baseline Model Selection

#### • Baseline Model: RT-DETR
- **Paper**: *DETRs Beat YOLOs on Real-time Object Detection* (CVPR 2024).
- RT-DETR bridges the gap between DETR (Detection Transformer) and YOLO, focusing on both accuracy and inference speed for real-time applications.

#### **Key Innovations**:
1. **Parallel Decoder Architecture**: Reduces latency compared to sequential decoders in traditional DETR models.
2. **Dynamic Query Design**: Adjusts the number of queries dynamically for better computational efficiency.
3. **Multi-scale Feature Fusion**: Enhances detection performance, especially for small objects.
4. **End-to-end Pipeline**: Avoids the need for complex post-processing, such as NMS (Non-Maximum Suppression).

#### **Performance**:
- Achieves state-of-the-art mAP (46.5) with 20M parameters and 60B FLOPs, outperforming YOLO models in real-time scenarios.

#### **Performance Comparison**
| Model       | mAP  | Parameters | FLOPs   |
|-------------|-------|------------|---------|
| RT-DETR     | 46.5  | 20M        | 60B     |
| Ours-v3     | 48.0  | 17M        | 23B     |
| Ours-v2     | 44.4  | 13M        | 20B     |
| Ours-v1     | 43.7  | 13M        | 20B     |
| Ours-v0     | 41.2  | 5.2M       | 6.4B    |

---

### 2. Model Compression and Optimization

#### • Compression Techniques
1. **Feature Map Reuse**:
   - Reused initial feature maps to optimize FLOPs while preserving accuracy.
   - Improved mAP from $43.7$ (Ours-v1) to $44.4$ (Ours-v2).
2. **Parameter Reduction**:
   - Reduced parameters from $20M$ (RT-DETR) to $5.2M$ (Ours-v0), significantly decreasing computation costs.

#### **Comparative Model Performance**
| Model       | mAP    | Parameters | FLOPs  |
|-------------|--------|------------|--------|
| RT-DETR     | 46.5   | 20M        | 60B    |
| Ours-v3     | 48.0   | 17M        | 23B    |
| Ours-v2     | 44.4   | 13M        | 20B    |
| Ours-v1     | 43.7   | 13M        | 20B    |
| Ours-v0     | 41.2   | 5.2M       | 6.4B   |

---

### 3. Deployment Optimization

#### • TensorRT Conversion
- Converted optimized models to **TensorRT**, achieving faster inference speeds on Nvidia Orin while maintaining acceptable accuracy levels.

#### • DeepStream Pipeline
- Integrated optimized models into a **DeepStream-based pipeline** for end-to-end deployment.

#### **Hardware: Nvidia Orin**
- **Specifications**:
  - GPU: Up to 2048-core Ampere with Tensor Cores.
  - RAM: Up to 32 GB LPDDR5x.
  - Optimized for high-performance, low-power inference tasks.

---

### 4. RT-DETR Contributions in Detail

#### **Addressing DETR's Challenges**:
1. **Decoder Bottleneck**:
   - Traditional DETR models use a sequential decoding process, increasing latency. RT-DETR introduces a parallel decoder for faster computation.
2. **Slow Convergence**:
   - DETR models require extensive training iterations due to bipartite matching. RT-DETR improves training efficiency with dynamic query updates.
3. **Inefficient Small Object Detection**:
   - Through multi-scale feature fusion, RT-DETR achieves better detection performance on smaller objects compared to DETR.

#### **Advantages Over YOLO**:
1. **Simplified Pipeline**:
   - Unlike YOLO, RT-DETR eliminates post-processing like NMS, reducing computational overhead.
2. **Higher mAP at Similar Speeds**:
   - RT-DETR provides superior mAP while maintaining inference speeds competitive with YOLO models.

---

### 5. Achievements

1. **Improved Model Performance**:
   - Developed Ours-v3, achieving $48.0$ mAP with $17M$ parameters and $23B$ FLOPs, surpassing RT-DETR's baseline.
2. **Efficient Nvidia Orin Deployment**:
   - Reduced FLOPs to $6.4B$ (Ours-v0) for efficient real-time inference.
3. **Seamless Integration**:
   - Optimized pipelines using TensorRT and DeepStream for robust deployment.

---

### References
1. RT-DETR: *DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024.
2. Nvidia TensorRT Documentation.
3. DeepStream SDK for Object Detection and Deployment.
