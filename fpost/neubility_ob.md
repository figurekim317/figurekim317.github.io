---
layout: fpost
title: "Project 1"
permalink: /fpost/1/
author: Dohyeong Kim
tags:
  - RGA
  - Object detection
  - Model Compression
---

### • Goal
#### \- Build a model that can perform stable computer vision tasks in an environment with limited resources
#### \- Refactor existing legacy code
#### \- Study models capable of instance segmentation using segmentation datasets


### 1. Object Detection: Network Design and Optimization

- **NVIDIA Xavier Resource**
  - CPU: 8-core ARM v8.2 64-bit CPU (Custom NVIDIA Carmel)
  - GPU: Volta Architecture with 512 NVIDIA CUDA cores and 64 Tensor Cores
  - RAM: 16 GB 256-bit LPDDR4x memory
  - USB: 4 USB 3.1 Gen 1 ports, 1 USB 3.1 Gen 2 port, 1 USB-C port
  - Camera: 2 MIPI CSI-2 D-PHY lanes, up to 16 simultaneous cameras

#### • YOLOv5 Model Performance Improvement and Quantization
- Conducted quantization to improve performance and lightweight design based on YOLOv5 models.
  - New developments in YOLOv5 increased model accuracy and speed on GPUs but introduced complexity for CPU deployment.
  - Compound scaling led to smaller, memory-bound networks such as YOLOv5s and larger, compute-bound networks such as YOLOv5l.
  - Post-processing and Focus blocks slowed down YOLOv5l, especially at larger input sizes.
  - Deployment performance differed significantly between GPUs and CPUs.
    - At batch size 1 and 640×640 input size, a T4 FP16 GPU instance on AWS (PyTorch) was over 7× faster than a 24-core C5 CPU instance on AWS (ONNX Runtime).

<figure>
  <div style="text-align:center">
    <div style="display:inline-block; margin-right:10px;">
      <img src="\fpost\images\pf1\yolov5 capture0.png" alt="Placeholder text" style="width:100%;">
    </div>
    <div style="display:inline-block;">
      <img src="\fpost\images\pf1\yolov5 capture1.png" alt="Placeholder text" style="width:100%;">
    </div>
  </div>
  <figcaption style="text-align:center">Fig 1. YOLOv5 result</figcaption>
</figure>

<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\transfer-learning-wandb-chart-1024x512.png" alt="Placeholder text" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 2. Transfer learning results on the VOC dataset for YOLOv5 models</figcaption>
</figure>


#### • Class Imbalance Problem Solution
- Solved the class imbalance problem by incorporating a numerical measure of data overlap (effective number) in the loss function.

<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\class imbalanced problem.png" alt="Placeholder text" style="width:70%;">
  </div>
  <figcaption style="text-align:center">
    Fig 3. (a) The common training pipeline of a generic detection network, consisting of 3 phases (feature extraction, detection and BB matching, labeling and sampling). (b) Illustration of example imbalance problems for object detection in each stage of the pipeline.
  </figcaption>
</figure>


### 2. Segmentation Model: Network Design and Optimization

#### • Model Exploration and Adaptation
- Utilized various models such as DDRNet, DeepLab V3+, and ESPNet.
- Optimized models for autonomous driving by lightweight design and structural enhancements.

<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\segmentation 0.png" alt="Placeholder text" style="width:70%;">
  </div>
  <figcaption style="text-align:center">
    Fig 4. Common failure modes for semantic segmentation regarding inference scale. Top row: the thin posts are inconsistently segmented in the downscaled (0.5×) image but are better predicted in the upscaled (2.0×) version. Bottom row: the large road/divider region is better segmented at lower resolution (0.5×).
  </figcaption>
</figure>

#### • Model Comparison

_Table 1: Performance comparison of several semantic segmentation models on the Cityscapes dataset._

| Model        | Model Size | Inference Speed (FPS) | Accuracy (Cityscapes) |
|--------------|------------|-----------------------|-----------------------|
| LiteSeg      | 1.2 MB     | 88.2                  | 70.6 mIoU             |
| EfficientPS  | 4.3 MB     | 23.5                  | 72.3 mIoU             |
| FastDepthSeg | 1.9 MB     | 140.8                 | 70.1 mIoU             |
| DDRNet       | 7.5 MB     | 82.7                  | 78.2 mIoU             |
| DeepLab V3+  | 8.4 MB     | 37.6                  | 77.7 mIoU             |


<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\segmentation 1.png" alt="Placeholder text" style="width:60%;">
  </div>
  <figcaption style="text-align:center">
    The approach by <a href="https://arxiv.org/abs/1511.03339">Chen, et al.</a> learns a dense attention mask for a fixed set of scales and fuses them into the final semantic prediction.
  </figcaption>
</figure>

<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\segmentation 2.png" alt="Placeholder text" style="width:80%;">
  </div>
  <figcaption style="text-align:center">
    Our hierarchical multi-scale attention method. Top: During training, the model learns to predict attention between adjacent scale pairs. Bottom: Inference is performed in a chained/hierarchical way to combine multiple scales. Lower-scale attention determines the contribution of the next higher scale.
  </figcaption>
</figure>


### 3. Model Quantization and Comparison
- Applied static quantization, dynamic quantization, and quantization-aware training.
- Compared performance across different models.

#### • Quantization Results

```html
<style>
table {
  margin-left: auto;
  margin-right: auto;
  border-collapse: collapse;
  width: 70%;
}
th, td {
  border: 1px solid black;
  padding: 8px;
  text-align: center;
}
th {
  background-color: #f2f2f2;
}

td:nth-child(1) {
  width: 10%;
}
td:nth-child(2) {
  width: 30%;
}
td:nth-child(3) {
  width: 30%;
}
td:nth-child(4) {
  width: 30%;
}
</style>
```
**Table 2. Performance comparison of different models under different quantization methods.**

| Model         | Quantization-aware Training     | Post Static Quantization      | Post Dynamic Quantization     |
|---------------|---------------------------------|--------------------------------|--------------------------------|
| **Metrics**   | **mIoU / Model Size / FPS**     | **mIoU / Model Size / FPS**   | **mIoU / Model Size / FPS**   |
| LiteSeg       | 65.1 / 0.6 MB / 172.4          | 62.8 / 0.5 MB / 194.5         | 64.0 / 0.6 MB / 178.7         |
| EfficientPS   | 68.7 / 2.15 MB / 45.9          | 65.3 / 1.7 MB / 57.1          | 67.0 / 2.15 MB / 51.3         |
| FastDepthSeg  | 67.0 / 0.95 MB / 274.6         | 64.8 / 0.8 MB / 294.8         | 65.8 / 0.95 MB / 281.3        |
| DDRNet        | 73.4 / 3.75 MB / 160.2         | 71.0 / 3.2 MB / 185.3         | 72.4 / 3.75 MB / 171.4        |
| DeepLab V3+   | 69.7 / 4.2 MB / 72.9           | 67.5 / 3.5 MB / 87.4          | 68.8 / 4.2 MB / 79.6          |

---

**Table 3: Results of quantization-aware training and post-training quantization on various models. Each cell shows mIoU / Model Size Decrease (%) / RAM Usage Decrease (%).**

| Model         | Quantization-aware Training           | Post Static Quantization            | Post Dynamic Quantization           |
|---------------|---------------------------------------|--------------------------------------|--------------------------------------|
| LiteSeg       | -4.3 / 16.7 / 0                      | -2.6 / 16.7 / -7                    | -1.8 / 0 / -5                       |
| EfficientPS   | -4.8 / 15.5 / -17                    | -4.8 / 20.9 / -7                    | -2.6 / 15.5 / -12                   |
| FastDepthSeg  | -2.2 / 2.3 / 42                      | -3.3 / 10.5 / 14                    | -1.9 / 2.3 / 23                     |
| DDRNet        | -3.3 / 9.7 / 44                      | -3.3 / 13.3 / 22                    | -1.8 / 9.7 / 36                     |
| DeepLab V3+   | -2.7 / 19 / 56                       | -2.9 / 16.7 / 38                    | -1.6 / 19 / 48                      |

---

### • ONNX Conversion and Optimization
- Performed ONNX conversion and various optimizations for more efficient deployment.

---

### • TensorRT Optimization for NVIDIA Xavier Environment
- Optimized models via TensorRT for inference on NVIDIA Xavier devices.
- Achieved significant improvements in inference speed while maintaining acceptable accuracy levels.

---

### 4. Legacy Code Refactoring

#### • GStreamer Pipeline Optimization
- Optimized pipelines based on GStreamer.

#### • Decoding Improvement
- Replaced software decoding with hardware decoding to reduce CPU overhead.

#### • Sensor Input and Modularization
- Routed various sensor inputs to the hardware decoding module.
- Modularized GStreamer and improved the deep learning model loading process using a plugin loader.

---

### 5. Weakly Supervised Instance Segmentation Model Development
- Developed a weakly supervised instance segmentation model using only semantic segmentation data.

---

### 6. Real-Time Object Detection Baseline Model and Multi-stage Optimizations

#### • Constraints and Baseline Model Selection
- Must operate in **real-time**.  
- Must run on a **Jetson platform** (e.g., Xavier).

With these constraints in mind, we chose as our baseline model from **CVPR 2024**:  
**RT-DETR (DETRs Beat YOLOs on Real-time Object Detection)**  

RT-DETR employs a DETR-like architecture optimized for real-time inference, offering competitive accuracy and speed compared to the YOLO family.  
- **Server GPU environment** already performs real-time inference.  
- **Jetson environment** has limited resources → additional model compression/optimization needed.

---

#### • Original vs. Lightweight Model (Ours-v0)

| Category                    | Model Size (Params) | FLOPs  | mAP   |
|-----------------------------|---------------------:|-------:|------:|
| **Original**                | 20M                 | 60B    | 46.5  |
| **Lightweight (Ours-v0)**  | 5.2M                | 6.4B   | 41.2  |

- **Original RT-DETR**  
  - Parameters: ~20M  
  - FLOPs: ~60B  
  - Accuracy: mAP 46.5  

- **Lightweight (Ours-v0)**  
  - Parameters: 5.2M  
  - FLOPs: 6.4B  
  - Accuracy: mAP 41.2  

Ours-v0 drastically reduces computational requirements and parameter counts while still achieving 41.2 mAP.

---

#### • Comparison with the YOLO Series

| Model     | mAP              | #Params  | FLOPs   |
|-----------|------------------|---------:|--------:|
| **Ours-v0** | **41.2**         | 5.2M     | 6.4B     |
| YOLOv5    | 37.4 (**-9.2%**) | 7.2M     | 16.5B   |
| YOLOv7    | 38.7 (**-6.1%**) | 6.2M     | 13.8B   |
| YOLOv8    | 37.3 (**-9.5%**) | 3.2M     | 8.7B    |

- Despite significant reductions in parameters and FLOPs,
- It outperforms or matches the YOLO series in terms of mAP.

---

#### • Further Model Revisions

Beyond Ours-v0 (5.2M params), we introduced additional optimizations to improve accuracy while adjusting computational cost:

- **Re-using the initial feature map**: improved Ours-v1 (43.7 mAP) → Ours-v2 (44.4 mAP).
- Further structural refinements, parameter count increased to 17M, leading to Ours-v3 with 48.0 mAP.

| Model    | mAP            | #Params | FLOPs |
|----------|----------------|--------:|------:|
| RT-DETR  | 46.5           | 20M     | 60B   |
| Ours-v0  | 41.2           | 5.2M    | 6.4B  |
| Ours-v1  | 43.7           | 13M     | 20B   |
| Ours-v2  | 44.4           | 13M     | 20B   |
| Ours-v3  | **48.0 (+3%)** | 17M     | 23B   |

Ours-v3 achieves 48.0 mAP, exceeding RT-DETR’s 46.5, while using fewer parameters (17M vs. 20M) and fewer FLOPs (23B vs. 60B).

---

#### • Inference & Pipeline Optimizations

1. **Model Computation Optimization (TensorRT conversion)**  
   - Converted trained models to ONNX and optimized with TensorRT for faster inference on Jetson (Xavier, etc.).  
   - Tested FP16/INT8 modes to maximize inference speed.

2. **Pipeline Optimization (DeepStream)**  
   - Utilized NVIDIA’s DeepStream SDK to optimize the end-to-end pipeline (input → decoding → inference → post-processing).  
   - Rebuilt the GStreamer-based pipeline to fully leverage hardware acceleration and reduce memory transfer overhead.  
   - Achieved real-time performance (>30 FPS) on Jetson Xavier.

---

### Summary

- Chose **RT-DETR** as the baseline, aiming for **real-time** object detection on Jetson.  
- **Ours-v0**: 5.2M parameters, 6.4B FLOPs, 41.2 mAP.  
- **Ours-v3**: 17M parameters, 23B FLOPs, 48.0 mAP (surpassing RT-DETR’s 46.5).  
- Achieved **real-time** inference on Jetson Xavier through **TensorRT** and **DeepStream** optimization.  

By balancing computational cost and accuracy, we successfully deployed a high-performance, real-time object detection model under limited resource conditions.
