---
layout: fpost
title: "Robust Depth Super-Resolution for Low-Resolution and Noisy Maps"
permalink: /fpost/rga/depth_estimation/
author: Dohyeong Kim
tags:   
  - RGA
  - CVPR
  - Depth Estimation
  - Model Compression
  - Optimization
---

## Goal
- **Achieved 1st Place** in the [2024 CVPR AIS Depth Compression Challenge](https://ai4streaming-workshop.github.io).  
- Develop a **robust super-resolution method** for degraded and noisy depth maps.  
- Demonstrate a **highly efficient** approach capable of near real-time performance.

---

## 1. Problem Definition & Dataset Analysis

### • Context of the Challenge
- Low-resolution (LR) depth maps suffer from severe degradation and noise, making them unreliable for high-resolution depth reconstruction.
- Sole reliance on LR depth maps and input images fails to produce accurate depth predictions.

### • Key Observations
- **HR vs. LR Depth Comparison**: LR depth maps show both resolution loss and significant corruption.  
- **Noise Impact**: High levels of noise in LR depth maps significantly affect depth reconstruction accuracy.

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig1.png" alt="Degraded LR Depth Map" style="width:70%;">
  </div>
  <figcaption style="text-align:center">**Fig 1. Visual comparison showing noise and resolution loss in the LR depth map.**</figcaption>
</figure>

---

## 2. Proposed Model & Approach

### • Utilizing **Relative Depth (Depth Anything)**
- The **pre-trained ‘Depth Anything’** model extracts **relative depth**, used as a supplementary guide for depth super-resolution.
- Inputs include the **relative depth map**, **low-resolution (LR) depth map**, and the **input image**, which are concatenated to form a multi-channel input for the network.

---

### • U-Net-like Structure with Tailored Design
Our architecture is based on a **U-Net-inspired framework**, retaining its characteristic **encoder-decoder structure** and **skip connections**, with significant enhancements to address the unique challenges of noisy LR depth maps.

#### **Encoder**
- The encoder features **two parallel paths**:
  - **Relative Depth Path**: Processes features from the relative depth map.
  - **LR Depth Path**: Extracts features from the noisy LR depth map.
- Both paths employ **NAFNet blocks** instead of conventional convolutional blocks, enhancing feature extraction.

#### **Fusion Module**
- Features from both paths are fused using **Adaptive Instance Normalization (AdaIN)**:
  - The relative depth features are normalized to align with the distribution of the LR depth map, improving feature compatibility.

#### **Decoder**
- The decoder reconstructs the high-resolution (HR) depth map by leveraging fused features, with skip connections providing contextual information.

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig2.png" alt="Model Pipeline" style="width:90%;">
  </div>
  <figcaption style="text-align:center">**Fig 2. Model pipeline integrating Depth Anything and U-Net-like structure.**</figcaption>
</figure>

---

### • Detailed Architecture
- The encoder performs four stages of downsampling, extracting progressively finer features.
- At each stage, the **Fusion Module** combines and normalizes features from both encoder paths.
- The decoder restores the fused features to the target resolution through upsampling.

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig3.png" alt="Detailed U-Net Structure" style="width:90%;">
  </div>
  <figcaption style="text-align:center">**Fig 3. Detailed architecture showing encoder, fusion module, and decoder.**</figcaption>
</figure>

---

## 3. Implementation & Training

### • Loss Function
The total loss function is designed to optimize both pixel-level accuracy and edge sharpness:
$$
\mathcal{L}_{\text{total}} = \| D_{\text{SR}} - D_{\text{GT}} \| + \| \text{Sobel}(D_{\text{SR}}) - \text{Sobel}(D_{\text{GT}}) \|
$$
- **First Term**: Minimizes pixel-wise error between the super-resolved depth map ($D_{\text{SR}}$) and the ground truth depth map ($D_{\text{GT}}$).
- **Second Term**: Emphasizes edge preservation using the Sobel operator to align edge structures between the prediction and ground truth.

---

### • Dataset & Pre-Training
- **Pretrained Model**: Initialized on the **MVS-Synthetic Dataset**.
- **Depth Map Range**: HR and LR depth maps are normalized to $[0, 1]$.
- **Depth Clipping**: Depth values are capped at 300 for consistency.
- **Validation Set**: The last 100 samples of the training set are reserved for validation.

---

### • Training Settings
- **Batch Size**: 8  
- **Learning Rate**:  
  - Depth Anything model: $2 \cdot 10^{-6}$  
  - U-Net: $2 \cdot 10^{-4}$  
- **Epochs**: 500  
- **Hardware**: Single NVIDIA A6000 GPU (~3 days).  
- **Inference Speed**: ~24 FPS on RTX 3090.  
- **Parameters**: 29M  

---

## 4. Results & Conclusion

### • Enhanced Detail
- The model outperforms the Depth Anything baseline by reconstructing finer depth edges and sharper details.
<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig4.png" alt="Enhanced Details" style="width:70%;">
  </div>
  <figcaption style="text-align:center">**Fig 4. Depth reconstruction showing enhanced detail compared to the baseline.**</figcaption>
</figure>

---

### • Noise Robustness
- Maintains strong accuracy even when the LR input contains heavy noise, demonstrating its robustness.
<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig5.png" alt="Noise Robustness" style="width:70%;">
  </div>
  <figcaption style="text-align:center">**Fig 5. Robust depth reconstruction under noisy LR input.**</figcaption>
</figure>

---

### • Real-Time Feasibility
- Operates efficiently at ~24 FPS on an RTX 3090, ensuring suitability for near real-time applications.

---

### • Summary
By integrating the strengths of the Depth Anything model with a tailored U-Net-like architecture, our model achieves exceptional depth super-resolution performance. It demonstrates robustness to noise, high-resolution detail recovery, and real-time feasibility, making it a compelling solution for both academic and practical applications.
