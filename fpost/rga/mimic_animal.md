---
layout: fpost
title: "Advanced Reconstruction with RAC: Animatable Categories from Videos"
permalink: /fpost/rga/mimic_animal/
author: Dohyeong Kim
tags:   
  - 3D Reconstruction
  - RAC
  - Neural Rendering
  - Animatable Models
  - Differentiable Rendering
  - Morphology Code
  - Soft Deformation
  - Robotics
---

## Goal
- Present a comprehensive analysis of RAC (Reconstructing Animatable Categories), incorporating detailed mathematical modeling.
- Highlight its potential for reconstructing and animating 3D models of animals from in-the-wild videos.
- Showcase its application for robotics, especially quadrupeds, and emphasize the inclusion of morphology variations and deformable models.

---

### 1. Method Overview

Given monocular video recordings, RAC constructs **animatable 3D models** that encode:
1. **Instance-specific Morphology** (e.g., body size, shape).
2. **Time-varying Articulations and Deformations** (e.g., skeletal motion, soft tissue movement).
3. **Video-specific 3D Background Models** (e.g., environmental context).

The optimization leverages **differentiable rendering**, allowing seamless integration of 3D modeling with observed image data.


<figure>
  <div style="text-align:center">
    <img src="images/rac_workflow.png" alt="RAC Workflow" style="width:90%;">
  </div>
  <figcaption style="text-align:center">Fig 1. RAC pipeline for generating animatable 3D models from monocular videos.</figcaption>
</figure>

---

### 2. Between-Instance Variation

To capture morphological diversity across instances, RAC introduces the **morphology code** $ \beta $, which encodes both **shape** and **skeleton** variations.

#### 2.1 Canonical Shape Representation

Each 3D point $ X \in \mathbb{R}^3 $ is associated with properties predicted by MLPs:

$$
(d, c_t) = \text{MLP}_{\text{SDF}}(X, \beta, \omega_a),
$$

where:
- $ d \in \mathbb{R} $: Signed distance for surface representation.
- $ c_t \in \mathbb{R}^3 $: Color conditioned on appearance $ \omega_a $.
- $ \beta \in \mathbb{R}^{32} $: Morphology code.
- $ \omega_a \in \mathbb{R}^{64} $: Frame-specific appearance (e.g., shadows).

An additional canonical feature vector $ \psi \in \mathbb{R}^{16} $ is computed:

$$
\psi = \text{MLP}_{\psi}(X).
$$

#### 2.2 Skeleton Representation

Skeletons are defined with fixed **category-level topology** and video-specific joint locations:

$$
J = \text{MLP}_{J}(\beta) \in \mathbb{R}^{3 \times B},
$$

where:
- $ B $: Number of bones.
- $ J $: Instance-specific joint positions.

#### 2.3 Skinning Field

The skinning weights $ W \in \mathbb{R}^{B+1} $ are defined as:

$$
W = \sigma_{\text{softmax}} \big(d_{\sigma}(X, \beta, \theta) + \text{MLP}_{W}(X, \beta, \theta)\big),
$$

where:
- $ \theta $: Articulation vector.
- $ d_{\sigma} $: Mahalanobis distance from Gaussian bones.

#### 2.4 Stretchable Bone Deformation

Morphological variations (e.g., limb elongation) are modeled by stretching canonical shapes:

$$
T_{\beta}^{s} = W_{\beta} G_{\beta} T_{\beta},
$$

where:
- $ T_{\beta} $: Canonical shape.
- $ G_{\beta} $: Bone transformations.
- $ W_{\beta} $: Skinning weights.

---

### 3. Within-Instance Variation

#### 3.1 Time-Varying Articulation

Joint rotations $ Q $ are computed via an MLP:

$$
Q = \text{MLP}_{A}(\theta) \in \mathbb{R}^{3 \times B},
$$

where $ \theta \in \mathbb{R}^{16} $ encodes skeletal articulation. Bone transformations $ G $ are derived through **forward kinematics** and applied with dual quaternion blend skinning (DQB):

$$
D(\beta, \theta) = (W_{\beta} G) T_{\beta}^s.
$$

#### 3.2 Time-Varying Soft Deformation

To account for dynamic deformations (e.g., fur, muscles):

$$
D(\beta, \theta, \omega_d) = D(D(\beta, \theta), \omega_d),
$$

where $ \omega_d \in \mathbb{R}^{64} $ encodes frame-specific deformations. A **real-NVP** framework ensures invertibility of deformation fields.

---

### 4. Scene Model and Background Reconstruction

To handle segmentation inaccuracies, RAC integrates a **background NeRF** conditioned on a video-specific code $ \gamma $:

$$
(\sigma, c_t) = \text{MLP}_{\text{bg}}(X, v, \gamma),
$$

where:
- $ \sigma $: Density of the background.
- $ c_t $: Color of the background conditioned on the viewing direction $ v $.

Foreground and background are rendered jointly for robust segmentation refinement.

---

### 5. Loss Functions

#### 5.1 Reconstruction Loss

The primary loss compares rendered and observed images:

$$
\mathcal{L}_{\text{recon}} = \mathcal{L}_{\text{sil}} + \mathcal{L}_{\text{rgb}} + \mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{feat}}.
$$

#### 5.2 Regularization Terms

1. **Morphology Code Regularization**:

$$
\mathcal{L}_{\beta} = \| D(\beta_1, \theta_1, \omega_{d1}) - D(\beta_2, \theta_2, \omega_{d2}) \|^2.
$$

2. **Soft Deformation Regularization**:

$$
\mathcal{L}_{\text{soft}} = \| D(\beta, \theta, \omega_d) - D(\beta, \theta) \|^2.
$$

3. **Sinkhorn Divergence for Joint Alignment**:

$$
\mathcal{L}_{\text{sinkhorn}} = \text{SD}(T_{\beta}, J_{\beta}),
$$

where $ \text{SD} $ measures the divergence between the canonical surface and joint positions.

---

### 6. Results and Experiments

#### **Comparative Model Performance**

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>CD (cm) ↓</th>
      <th>F@2% ↑</th>
      <th>F@5% ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>HuMoR</td>
      <td>9.8</td>
      <td>47.5</td>
      <td>83.7</td>
    </tr>
    <tr>
      <td>ICON</td>
      <td>10.1</td>
      <td>39.9</td>
      <td>85.2</td>
    </tr>
    <tr>
      <td>BANMo</td>
      <td>9.3</td>
      <td>54.4</td>
      <td>85.5</td>
    </tr>
    <tr>
      <td><b>RAC</b></td>
      <td><b>6.0</b></td>
      <td><b>72.5</b></td>
      <td><b>94.4</b></td>
    </tr>
  </tbody>
</table>


<figure>
  <div style="text-align:center">
    <img src="images/rac_results.png" alt="RAC Results" style="width:90%;">
  </div>
  <figcaption style="text-align:center">Fig 2. RAC outperforms baselines in both coarse and fine detail reconstruction.</figcaption>
</figure>

---

### Conclusion

RAC achieves state-of-the-art performance in animatable 3D reconstruction by combining morphological modeling, soft deformations, and differentiable rendering. Its ability to extract lifelike motion from videos unlocks new possibilities for robotics and virtual reality applications.
