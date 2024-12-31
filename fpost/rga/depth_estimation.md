---
layout: fpost
title: "Project 6"
permalink: /fpost/rga/depth_estimation/
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
    <img src="\fpost\rga\depth_img\fig1.png" alt="Degraded LR Depth Map" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 1. Visual comparison showing noise and resolution loss in the LR depth map</figcaption>
</figure>

<br/>

## 2. Proposed Model & Approach

### • Utilizing **Relative Depth (Depth Anything)**
- We incorporate a **pre-trained ‘Depth Anything’** model to extract **relative depth**, which serves as an additional guide for upsampling.
- The **relative depth map**, along with the **low-resolution (LR) depth map** and the **input image**, are concatenated to form multi-channel inputs. This combined input enriches the model with supplementary features necessary for depth super-resolution.

---

### • U-Net-like Structure with Tailored Design
Our architecture is inspired by the **U-Net framework**, retaining its characteristic **encoder-decoder design** and **skip connections**. However, the structure has been customized to address specific challenges in depth super-resolution:

#### - **Encoder**
- Features two parallel paths:
  - **Relative Depth Path**: Processes features from the extracted relative depth map.
  - **LR Depth Path**: Handles features from the degraded low-resolution depth map.
- Both paths use **NAFNet blocks** instead of standard convolutional blocks to enhance feature extraction at each downsampling stage.

#### - **Fusion Module**
- Integrates features from the relative depth and LR depth paths using **Adaptive Instance Normalization (AdaIN)**.
- This normalization aligns the relative depth distribution to that of the LR depth map, ensuring compatibility and improving fusion quality.

#### - **Decoder**
- Reconstructs the final **super-resolution (SR) depth map** by leveraging the fused features, with **skip connections** providing additional contextual information.

---

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig2.png" alt="Model Pipeline" style="width:70%;">
  </div>
  <figcaption style="text-align:center">**Fig 2. Model pipeline integrating Depth Anything and U-Net-like structure.**</figcaption>
</figure>

---

### • Detailed Architecture
- The encoder performs four stages of downsampling, extracting progressively finer features from both the relative depth and LR depth inputs. At each stage:
  - The extracted features are normalized and combined in the **Fusion Module**.
  - The **decoder** then upscales the fused features stage by stage, restoring them to the target resolution.

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\depth_img\fig3.png" alt="Detailed U-Net Structure" style="width:70%;">
  </div>
  <figcaption style="text-align:center">**Fig 3. Detailed architecture showing encoder, fusion module, and decoder.**</figcaption>
</figure>

---

### • Summary
By tailoring the U-Net architecture to integrate relative depth as an additional guide, and employing **NAFNet blocks** and **ADAIN-based normalization**, the model effectively addresses the challenges of degraded low-resolution depth maps. This approach combines the strengths of the U-Net framework with task-specific enhancements, achieving superior depth super-resolution performance.


<br/>

## 3. Implementation & Training

### • Loss Function
Our loss function is designed to optimize both pixel-level accuracy and edge sharpness in the depth prediction:
$$
\mathcal{L}_{\text{total}} = \| D_{\text{SR}} - D_{\text{GT}} \| + \| \text{Sobel}(D_{\text{SR}}) - \text{Sobel}(D_{\text{GT}}) \|
$$

- **First Term**: L1 Loss minimizes pixel-wise error between the super-resolved depth map (\(D_{\text{SR}}\)) and the ground truth depth map (\(D_{\text{GT}}\)).
- **Second Term**: Sobel-based Edge Loss emphasizes and preserves edge details by aligning the edge structures of the predicted and ground truth depth maps.

---

### • Dataset & Pre-Training
To ensure robust model performance, the training process involves carefully curated datasets:
- **Pretrained Model**: Pre-trained on the **MVS-Synthetic Dataset** for initialization.
- **Depth Map Range**: Both HR and LR depth maps are normalized to the range \([0, 1]\).
- **Depth Clipping**: Maximum depth values of HR and LR depth maps are clipped to 300 to maintain consistency with the challenge dataset.
- **Validation Set**: The last 100 samples of the training dataset are reserved for validation.

---

### • Training Settings
- **Batch Size**: 8
- **Learning Rate**:
  - Depth Anything model: \(2 \times 10^{-6}\)
  - U-Net: \(2 \times 10^{-4}\)
- **Epochs**: 500 epochs
- **Hardware**: Single NVIDIA A6000 GPU, requiring approximately 3 days of training.
- **Inference Speed**: The model achieves an inference speed of approximately **24 FPS** on an RTX 3090.
- **Number of Parameters**: 29 million

---

### • Summary
The proposed model is rigorously trained and evaluated on synthetic and real-world datasets, employing an efficient loss function to optimize both pixel accuracy and edge preservation. By leveraging high-performance hardware and robust datasets, the model delivers high-quality depth super-resolution with real-time inference capability.


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