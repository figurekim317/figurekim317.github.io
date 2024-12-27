---
layout: fpost
title: "Project 6"
permalink: /fpost/6/
author: Dohyeong Kim
tags:   
  - RGA
  - CVPR
  - Depth estimation
  - Model Compression
  - Optimization
---

### • Goal
- **Achieved 1st Place** in the [2024 CVPR AIS Depth Compression Challenge](https://ai4streaming-workshop.github.io)  
- Develop a **robust super-resolution method** for degraded and noisy depth maps  
- Demonstrate a **highly efficient** approach capable of near real-time performance

<br/>

### 1. Problem Definition & Dataset Analysis

- **Context of the Challenge**  
  - Low-resolution (LR) depth maps display severe degradation and noise, resulting in significant reliability issues.  
  - Relying solely on the LR depth map and the input image is insufficient for accurate depth upsampling.

- **Key Observations**  
  - **HR vs. LR Depth Comparison**: LR depth maps suffer not only from resolution loss but also from heavy corruption.  
  - **Noise Impact**: The substantial noise in LR depth maps undermines depth reconstruction quality.

<figure>
  <div style="text-align:center">
    <!-- Example placeholder -->
    <img src="images/depth_challenge_example1.png" alt="Degraded LR Depth Map" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 1. Visual comparison showing noise and resolution loss in the LR depth map.</figcaption>
</figure>

<br/>

### 2. Proposed Model & Approach

#### • Utilizing **Relative Depth (Depth Anything)**
- Use a **pre-trained ‘Depth Anything’** model to extract **relative depth** as an additional guide.
- Concatenate the relative depth with both the LR depth map and the input image, forming multi-channel inputs for the network.

#### • U-Net with Separate Paths
- **U-Net Architecture**:
  - **Encoder**: Splits into two paths — one for **relative depth** and one for **LR depth**.
  - **Fusion Module**: Combines features with **Adaptive Instance Normalization (AdaIN)**.
  - **Decoder**: Leverages skip connections for final high-resolution depth prediction.

<figure>
  <div style="text-align:center">
    <!-- Example placeholder -->
    <img src="images/unet_structure.png" alt="U-Net Architecture" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 2. U-Net structure with dual input streams and a fusion module.</figcaption>
</figure>

<br/>

### 3. Implementation & Training

#### • Loss Functions
- **L1 Loss**: Primary objective minimizing pixel-wise error.  
- **Sobel-based Edge Loss**: Preserves and accentuates edge details in depth predictions.

#### • Dataset & Pre-Training
- **MVS-Synthetic Dataset**: Utilized for initial pre-training before finalizing with the challenge dataset.  
- **Depth Range Clipping**: Challenge dataset depths were capped at 300.

#### • Training Settings
- **Batch Size**: 8  
- **GPU**: Single A6000, ~3 days of training  
- **Inference Speed**: ~24 FPS on an RTX 3090

<figure>
  <div style="text-align:center">
    <!-- Example placeholder -->
    <img src="images/training_process.png" alt="Training pipeline" style="width:60%;">
  </div>
  <figcaption style="text-align:center">Fig 3. Overview of the training pipeline with depth clipping and pre-training.</figcaption>
</figure>

<br/>

### 4. Key Challenges & Solutions

- **Noisy LR Depth Input**  
  - Sole reliance on LR depth often yields blurred or incorrect depth outputs.  
  - **Solution**: Introduce **relative depth** as an additional guidance channel.

- **Distribution Discrepancy**  
  - Relative depth vs. ground truth distributions differ significantly.  
  - **Solution**: **AdaIN** aligns relative depth features to the LR depth distribution, closer to true depth’s scale.

- **Preserving Fine Details**  
  - Plain L1 loss may fail to capture subtle boundaries.  
  - **Solution**: Incorporate **Edge Loss** via Sobel to focus on edges.

<br/>

### 5. Weakly Supervised Extensions & Future Work

- **Potential for Weakly Supervised Instance Segmentation**  
  - Relative depth could be leveraged for instance-level segmentation where annotated data is limited.  
  - Combining **depth cues** with partial labels could boost instance segmentation pipelines.

<figure>
  <div style="text-align:center">
    <!-- Example placeholder -->
    <img src="images/instance_segmentation_future.png" alt="Weakly supervised instance segmentation" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 4. Concept diagram for extending depth-based supervision to instance segmentation.</figcaption>
</figure>

<br/>

### 6. Results & Conclusion
- **Enhanced Detail**  
  - Outperforms using only the Depth Anything model by yielding more precise depth edges and details.
- **Noise Robustness**  
  - Maintains strong accuracy despite heavy noise in the LR depth input.
- **Real-Time Feasibility**  
  - Operates at ~24 FPS on an RTX 3090, demonstrating near real-time performance.

<figure>
  <div style="text-align:center">
    <!-- Example placeholder -->
    <img src="images/final_result.png" alt="Final Depth Results" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Fig 5. Comparison highlighting improved details and fewer artifacts in the final predictions.</figcaption>
</figure>