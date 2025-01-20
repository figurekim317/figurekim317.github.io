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

### 1. Problem Definition & Dataset Analysis

#### Context of the Challenge
- Low-resolution (LR) depth maps suffer from degradation and noise, making them unreliable for upsampling:
  - Given a low-resolution depth map $D_{\text{LR}}$ and a corresponding RGB input $I$, the goal is to reconstruct a high-resolution depth map $D_{\text{SR}}$ such that:
    $$
    D_{\text{SR}} \approx D_{\text{GT}}
    $$
    where $D_{\text{GT}}$ is the ground truth depth map.

#### Key Observations
- **Resolution Degradation**: $D_{\text{LR}}$ suffers from downsampling artifacts and spatial corruption:
  $$
  D_{\text{LR}} = \downarrow (D_{\text{GT}}) + \eta
  $$
  where $\downarrow$ denotes downsampling and $\eta$ is additive noise.
- **Noise Impact**: The added noise $\eta$ significantly disrupts depth reconstruction quality.

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig1.png" alt="Degraded LR Depth Map" style="width:60%;">
  </div>
  <figcaption style="text-align:center">Fig 1. Visual comparison showing noise and resolution loss in the LR depth map.</figcaption>
</figure>

---

### 2. Proposed Model & Approach

#### Utilizing Relative Depth (Depth Anything)
- The **pre-trained ‘[Depth Anything](https://depth-anything.github.io)’** model extracts **relative depth**, used as a supplementary guide for depth super-resolution:
  $$
  D_{\text{Rel}} = f_{\text{DepthAnything}}(I)
  $$
- Inputs include the **relative depth map**, **low-resolution (LR) depth map**, and the **input image**, which are concatenated to form a multi-channel input:
  $$
  X = [D_{\text{Rel}}, D_{\text{LR}}, I]
  $$
  where $[\cdot]$ denotes concatenation.

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\depthanything_pipeline.png" alt="Depth anything pipeline" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Fig 2. Depth anything pipeline.</figcaption>
</figure>

---

#### U-Net-like Structure with Tailored Design
Our architecture is based on a **U-Net-inspired framework**, retaining its characteristic **encoder-decoder structure** and **skip connections**, with significant enhancements to address the unique challenges of noisy LR depth maps.

#### **Encoder**
- The encoder processes the multi-channel input $X$ through **two parallel paths**:
  - **Relative Depth Path**:
    $$
    F_{\text{Rel}} = \text{NAFNet}(D_{\text{Rel}})
    $$
  - **LR Depth Path**:
    $$
    F_{\text{LR}} = \text{NAFNet}(D_{\text{LR}})
    $$
- Features $F_{\text{Rel}}$ and $F_{\text{LR}}$ are progressively downsampled across multiple levels using [**NAFNet blocks**](https://arxiv.org/pdf/2204.04676), enhancing feature extraction and representation.

#### **Fusion Module**
- Features from the relative depth and LR depth paths are fused using **Adaptive Instance Normalization (AdaIN)** to align their distributions:
  $$
  F_{\text{Fusion}} = \text{AdaIN}(F_{\text{Rel}}, F_{\text{LR}})
  $$
  where:
  $$
  \text{AdaIN}(F_{\text{Rel}}, F_{\text{LR}}) = \sigma_{\text{LR}} \cdot \frac{F_{\text{Rel}} - \mu_{\text{Rel}}}{\sigma_{\text{Rel}}} + \mu_{\text{LR}}
  $$
  $\mu$ and $\sigma$ denote the mean and variance of the features, respectively.

#### **Decoder**
- The decoder reconstructs the high-resolution (HR) depth map by progressively upsampling the fused features $F_{\text{Fusion}}$, while utilizing skip connections from the encoder:
  $$
  D_{\text{SR}} = \text{Decoder}(F_{\text{Fusion}})
  $$

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig2.png" alt="Model Pipeline" style="width:90%;">
  </div>
  <figcaption style="text-align:center">Fig 3. Model pipeline integrating Depth Anything and U-Net-like structure.</figcaption>
</figure>

---

#### Detailed Architecture
- The encoder performs **four stages of downsampling**, progressively extracting finer features from the input.
- At each stage, the **Fusion Module** normalizes and combines features from the two encoder paths.
- The decoder restores the fused features to the target resolution via upsampling, integrating skip connections for enhanced reconstruction quality.

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig3.png" alt="Detailed U-Net Structure" style="width:90%;">
  </div>
  <figcaption style="text-align:center">Fig 4. Detailed architecture showing encoder, fusion module, and decoder.</figcaption>
</figure>

---

### 3. Implementation & Training

#### Loss Function
The total loss function consists of a pixel-level reconstruction term and an edge preservation term:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pixel}} + \mathcal{L}_{\text{edge}}
$$
- **Pixel Reconstruction Loss**:
  $$
  \mathcal{L}_{\text{pixel}} = \| D_{\text{SR}} - D_{\text{GT}} \|_1
  $$
- **Edge Preservation Loss**:
  $$
  \mathcal{L}_{\text{edge}} = \| \text{Sobel}(D_{\text{SR}}) - \text{Sobel}(D_{\text{GT}}) \|_1
  $$

---

#### Dataset & Pre-Training
- **Pretrained Model**: Initialized on the **MVS-Synthetic Dataset**:
  $$
  D_{\text{LR}}, D_{\text{GT}} \in [0, 1]
  $$
- **Depth Clipping**: Depth values are clipped to:
  $$
  D_{\text{GT, max}} = 300
  $$
- **Validation Set**: Last 100 samples reserved for validation.

---

#### Training Settings
- **Batch Size**: 8  
- **Learning Rate**:
  $$
  \alpha_{\text{DepthAnything}} = 2 \cdot 10^{-6}, \quad \alpha_{\text{U-Net}} = 2 \cdot 10^{-4}
  $$
- **Epochs**: 500  
- **Hardware**: Single NVIDIA A6000 GPU (~3 days).  
- **Inference Speed**: ~24 FPS on RTX 3090.  
- **Parameters**: 29M  

---

### 4. Results & Conclusion

#### Enhanced Detail
- Achieves finer edge and detail reconstruction compared to the baseline:
  $$
  D_{\text{SR}}^{\text{Ours}} \approx D_{\text{GT}}, \quad D_{\text{SR}}^{\text{Baseline}} \ll D_{\text{GT}}
  $$

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig4.png" alt="Enhanced Details" style="width:90%;">
  </div>
  <figcaption style="text-align:center">Fig 5. Depth reconstruction showing enhanced detail compared to the baseline.</figcaption>
</figure>

---

#### Noise Robustness
- Effectively mitigates noise from $D_{\text{LR}}$, retaining high accuracy:
  $$
  \| D_{\text{SR}} - D_{\text{GT}} \| < \| D_{\text{LR}} - D_{\text{GT}} \|
  $$

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig5.png" alt="Noise Robustness" style="width:90%;">
  </div>
  <figcaption style="text-align:center">Fig 6. Robust depth reconstruction under noisy LR input.</figcaption>
</figure>

---

#### Real-Time Feasibility
- Operates efficiently at ~24 FPS on an RTX 3090, enabling real-time applications:
  $$
  \text{Speed}_{\text{Ours}} = 24 \, \text{FPS}
  $$

---

#### Summary
By leveraging $D_{\text{Rel}}$ from the Depth Anything model and integrating a tailored U-Net architecture, our approach achieves robust super-resolution. It effectively handles noise, reconstructs fine details, and operates in real-time, making it highly suitable for practical deployment.