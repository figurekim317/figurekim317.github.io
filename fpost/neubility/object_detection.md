---
layout: fpost
title: "Object detection inference on device"
permalink: /fpost/neubility/object_detection/
author: Dohyeong Kim
tags:   
  - Autonomy driving
  - Object detection
  - Segmentation
  - Model Compression
  - Optimization
---

### • Goal
#### \- Build a model that can perform stable computer vision tasks in an environment with limited resources
#### \- Refactoring of existing lagacy code
#### \- Instance segmentation possible model study using segmentation dataset
 

 
### 1. Object Detection: Network Design and Optimization

- \- Nvidia Xavier resource
    - CPU: 8-core ARM v8.2 64-bit CPU (Custom NVIDIA Carmel)
    - GPU: Volta Architecture with 512 NVIDIA CUDA cores and 64 Tensor Cores
    - RAM: 16 GB 256-bit LPDDR4x memory
    - USB: 4 USB 3.1 Gen 1 ports, 1 USB 3.1 Gen 2 port, 1 USB-C port
    - Camera: 2 MIPI CSI-2 D-PHY lanes, up to 16 simultaneous cameras

#### • YOLOv5 Model Performance Improvement and Quantization
- \- Conducted quantization for YOLOv5-based model performance improvement and lightweight design.
  - The new developments in YOLOv5 improved model accuracy and speed on GPUs, but added complexity for CPU deployments.
  - Compound scaling resulted in small, memory-bound networks such as YOLOv5s and larger, compute-bound networks such as YOLOv5l.
  - Post-processing and Focus blocks slowed down YOLOv5l, especially at larger input sizes.
  - Deployment performance between GPUs and CPUs was significantly different.
    - At batch size 1 and 640x640 input size, there was a more than 7x gap in performance between a T4 FP16 GPU instance on AWS running PyTorch and a 24-core C5 CPU instance on AWS running ONNX Runtime.

<figure>
  <div style="text-align:center">
    <div style="display:inline-block; margin-right:10px;">
      <img src="\fpost\images\pf1\yolov5 capture0.png" alt="대체 텍스트" style="width:100%;">
    </div>
    <div style="display:inline-block;">
      <img src="\fpost\images\pf1\yolov5 capture1.png" alt="대체 텍스트" style="width:100%;">
    </div>
  </div>
  <figcaption style="text-align:center">Fig 1. Yolov5 result</figcaption>
</figure>

<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\transfer-learning-wandb-chart-1024x512.png" alt="대체 텍스트" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 2. Transfer learning results on the VOC dataset for the YOLOv5 models</figcaption>
</figure>


#### • Class Imbalanced Problem Solution
- \- Solved class imbalance problem by reflecting the numerical measurement of data overlap (effective number) in the loss function.

<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\class imbalanced problem.png" alt="대체 텍스트" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 3. (a) The common training pipeline of a generic detection network. The pipeline has 3 phases (i.e. feature extraction, detection and BB matching, labeling and sampling) represented by different background colors. (b) Illustration of an example imbalance problem from each category for object detection through the training pipeline. Background colors specify at which phase an imbalance problem occurs.</figcaption>
</figure>

### 2. Segmentation Model: Network Design and Optimization

#### • Model Exploration and Adaptation
- \- Utilized various models such as DDRNet, DeepLab V3+, and ESPNet.
- \- Optimized models for autonomous driving situations through lightweight design and structural optimization.

<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\segmentation 0.png" alt="대체 텍스트" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 4. Common failures modes for semantic segmentation as they relate to inference scale. In the first row, the thin posts are inconsistently segmented in the scaled down (0.5x) image, but better predicted in the scaled-up (2.0x) image. In the second row, the large road / divider region is better segmented at lower resolution (0.5x).</figcaption>
</figure>


#### • Model Comparison

_Table 1: Performance comparison of several semantic segmentation models on the Cityscapes dataset._

| Model        | Model Size | Inference Speed (FPS) | Accuracy (Cityscapes dataset) |
|--------------|------------|-----------------------|-------------------------------|
| LiteSeg      | 1.2 MB     | 88.2                  | 70.6 mIoU                     |
| EfficientPS  | 4.3 MB     | 23.5                  | 72.3 mIoU                     |
| FastDepthSeg | 1.9 MB     | 140.8                 | 70.1 mIoU                     |
| DDRNet       | 7.5 MB     | 82.7                  | 78.2 mIoU                     |
| DeepLab V3+  | 8.4 MB     | 37.6                  | 77.7 mIoU                     |


<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\segmentation 1.png" alt="대체 텍스트" style="width:60%;">
  </div>
  <figcaption style="text-align:center">The explicit approach of <a href="https://arxiv.org/abs/1511.03339">Chen, et al.</a> learns a dense attention mask for a fixed set of scales to combine them to form a final semantic prediction.fusion.</figcaption>
</figure>
 

<figure>
  <div style="text-align:center">
    <img src="\fpost\images\pf1\segmentation 2.png" alt="대체 텍스트" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Our hierarchical multi-scale attention method. Top: During training, our model learns to predict attention between two adjacent scale pairs. Bottom: Inference is done in a chained/hierarchical manner in order to combine multiple scales of predictions together. Lower scale attention determines the contribution of the next higher scale.</figcaption>
</figure>



### 3. Model Quantization and Comparison
- \- Applied static quantization, dynamic quantization, and quantization-aware training.
- \- Compared the performance of different models.


#### • Quantization Results

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

_Table 2. Performance comparison of different models under different quantization methods._

| Model       | Quantization-aware Training  | Post Static Quantization  | Post Dynamic Quantization |
|-------------|:----------------------------:|:-------------------------:|:--------------------------:|
| Metrics            | mIoU / Model Size / FP    S  | mIoU / Model Size / FPS    | mIoU / Model Size / FPS |
|-------------|-------------------------------|------------------------------|-------------------------------|
| LiteSeg     | 65.1 / 0.6 MB / 172.4         | 62.8 / 0.5 MB / 194.5        | 64.0 / 0.6 MB / 178.7          |
| EfficientPS | 68.7 / 2.15 MB / 45.9         | 65.3 / 1.7 MB / 57.1         | 67.0 / 2.15 MB / 51.3          |
| FastDepthSeg| 67.0 / 0.95 MB / 274.6        | 64.8 / 0.8 MB / 294.8        | 65.8 / 0.95 MB / 281.3         |
| DDRNet      | 73.4 / 3.75 MB / 160.2        | 71.0 / 3.2 MB / 185.3        | 72.4 / 3.75 MB / 171.4         |
| DeepLab V3+ | 69.7 / 4.2 MB / 72.9          | 67.5 / 3.5 MB / 87.4         | 68.8 / 4.2 MB / 79.6           |


_Table 3: Results of quantization-aware training and post-training quantization on various models. Each cell shows mIoU / Model Size Decrease (%) / RAM Usage Decrease (%)._

| -------------|:-----------------------------:|:---------------------------:|:----------------------------:|
|  Metrics     | mIoU / Model Size Decrease (%) / RAM Usage Decrease (%)  | mIoU / Model Size Decrease (%) / RAM Usage Decrease (%) | mIoU / Model Size Decrease (%) / RAM Usage Decrease (%) |
| -------------|-------------------------------|------------------------------|-------------------------------|
| LiteSeg       | -4.3 / 16.7 / 0               | -2.6 / 16.7 / -7             | -1.8 / 0 / -5                 |
| EfficientPS   | -4.8 / 15.5 / -17             | -4.8 / 20.9 / -7             | -2.6 / 15.5 / -12             |
| FastDepthSeg  | -2.2 / 2.3 / 42               | -3.3 / 10.5 / 14             | -1.9 / 2.3 / 23               |
| DDRNet        | -3.3 / 9.7 / 44               | -3.3 / 13.3 / 22             | -1.8 / 9.7 / 36               |
| DeepLab V3+   | -2.7 / 19 / 56                | -2.9 / 16.7 / 38             | -1.6 / 19 / 48                |





#### • ONNX Conversion and Optimization
- \- Performed ONNX conversion and optimization for better deployment

#### • TensorRT Optimization for Nvidia Xavier Environment
- \- Optimized models using TensorRT for inference on Nvidia Xavier devices
- \- Achieved significant improvement in inference speed while maintaining acceptable accuracy levels




### 4. Legacy Code Refactoring

#### • GStreamer Pipeline Optimization
- \- Optimized GStreamer-based pipeline

#### • Decoding Improvement
- \- Replaced SW decoding with HW decoding to reduce CPU overhead

#### • Sensor Input and Modularization
- \- Directed various sensor inputs to the HW decoding module
- \- Modularized GStreamer and improved the deep learning model loading process using plugin loader





### 5. Weakly Supervised Instance Segmentation Model Development

- \- Developed weakly supervised instance segmentation model using semantic segmentation data