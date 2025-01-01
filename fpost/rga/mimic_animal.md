---
layout: fpost
title: "Data-Driven Quadruped Locomotion with RAC: From Videos to Robots"
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
  - Reinforcement Learning
  - Domain Adaptation
  - Quadruped Robots
  - Data-Driven Locomotion

---

## Goal
- Present a comprehensive analysis of RAC (Reconstructing Animatable Categories), incorporating detailed mathematical modeling.
- Highlight its potential for reconstructing and animating 3D models of animals from in-the-wild videos.
- Showcase its application for robotics, especially quadrupeds, emphasizing the inclusion of morphology variations, deformable models, and sim-to-real adaptation.
- Emphasize a data-driven approach to obtaining locomotion data directly from videos, bypassing traditional motion capture setups.
- Bridge video-based motion data with reinforcement learning for robust quadruped locomotion.

---

### 1. Method Overview

Given monocular video recordings, RAC constructs **animatable 3D models** that encode:
1. **Instance-specific Morphology** (e.g., body size, shape).
2. **Time-varying Articulations and Deformations** (e.g., skeletal motion, soft tissue movement).
3. **Video-specific 3D Background Models** (e.g., environmental context).

The pipeline integrates the RAC reconstruction output into a quadruped robot's locomotion control system through **inverse kinematics (IK)** and **reinforcement learning (RL)**. This approach leverages video-based joint motion extraction instead of traditional motion capture methods to achieve natural and agile locomotion policies.

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\mimic_img\fig1.png" alt="RAC Workflow" style="width:90%;">
  </div>
  <figcaption style="text-align:center">Fig 1. RAC pipeline for generating animatable 3D models from monocular videos and retargeting motion to a quadruped robot.</figcaption>
</figure>

---

### 2. Between-Instance Variation

To capture morphological diversity across instances, RAC introduces the **morphology code** $\beta$, which encodes both **shape** and **skeleton** variations.

#### 2.1 Canonical Shape Representation

Each 3D point $X \in \mathbb{R}^3$ is associated with properties predicted by MLPs:

$$
(d, c_t) = \text{MLP}_{\text{SDF}}(X, \beta, \omega_a),
$$
where $d \in \mathbb{R}$ is the signed distance for surface representation, $c_t \in \mathbb{R}^3$ is the color conditioned on appearance $\omega_a$, $\beta \in \mathbb{R}^{32}$ is the morphology code, and $\omega_a \in \mathbb{R}^{64}$ represents frame-specific appearance (e.g., shadows).

An additional canonical feature vector $\psi \in \mathbb{R}^{16}$ is computed as:

$$
\psi = \text{MLP}_{\psi}(X).
$$

#### 2.2 Skeleton Representation

Skeletons are defined with fixed **category-level topology** and video-specific joint locations:

$$
J = \text{MLP}_{J}(\beta) \in \mathbb{R}^{3 \times B},
$$

where $B$ is the number of bones, and $J$ represents the instance-specific joint positions.

#### 2.3 Skinning Field

The skinning weights $W \in \mathbb{R}^{B+1}$ are defined as:

$$
W = \sigma_{\text{softmax}} \big(d_{\sigma}(X, \beta, \theta) + \text{MLP}_{W}(X, \beta, \theta)\big),
$$

where $\theta$ is the articulation vector, and $d_{\sigma}$ is the Mahalanobis distance from Gaussian bones.

#### 2.4 Stretchable Bone Deformation

Morphological variations (e.g., limb elongation) are modeled by stretching canonical shapes:

$$
T_{\beta}^{s} = W_{\beta} G_{\beta} T_{\beta},
$$

where $T_{\beta}$ represents the canonical shape, $G_{\beta}$ is the bone transformations, and $W_{\beta}$ refers to the skinning weights.

---

### 3. From Video to Robot

<figure>
  <div style="text-align:center">
    <img src="\fpost\rga\mimic_img\fig2.png" alt="RAC Pipeline" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 2. The framework consists of three stages: motion retargeting, motion imitation, and domain adaptation. It receives as input motion data recorded from an animal, and outputs a control policy that enables a real robot to reproduce the motion.</figcaption>
</figure>


#### 3.1 Video Input
- Collect single-view or multi-view videos of animal locomotion (e.g., dogs).

#### 3.2 RAC Reconstruction
- Obtain a **3D canonical model**, per-frame articulations $\theta$, and morphological differences $\Delta J_\beta$.

#### 3.3 Motion Retargeting
The output motion is mapped to the robot using inverse kinematics (IK):

$$
\min_{q_{0:T}} \sum_t \sum_i \| \hat{x}_i(t) - x_i(q_t) \|^2 + (\bar{q} - q_t)^T W (\bar{q} - q_t),
$$

where $\hat{x}_i(t)$ are target positions, $q_t$ are robot joint variables, and $W$ is a weighting matrix for joint constraints.

#### 3.4 Motion Imitation (RL)
In simulation (e.g., PyBullet, Mujoco), define a reward function to guide policy learning:

$$
r_t = w_p r_t^p + w_v r_t^v + w_e r_t^e + w_{rp} r_t^{rp} + w_{rv} r_t^{rv},
$$

where $r_t^p$, $r_t^v$, and $r_t^e$ evaluate pose, velocity, and end-effector accuracy, respectively:

- Pose Reward:
$$
r_t^p = \exp \left(-5 \sum_j \| \hat{q}_j(t) - q_j(t) \|^2 \right).
$$

- Velocity Reward:
$$
r_t^v = \exp \left(-0.1 \sum_j \| \dot{\hat{q}}_j(t) - \dot{q}_j(t) \|^2 \right).
$$

- End-Effector Reward:
$$
r_t^e = \exp \left(-40 \sum_e \| \hat{x}_e(t) - x_e(t) \|^2 \right).
$$

- Root Pose and Velocity Rewards:
$$
r_t^{rp} = \exp \left(-20 \| \hat{x}_{\text{root}}(t) - x_{\text{root}}(t) \|^2 \right),
$$
$$
r_t^{rv} = \exp \left(-2 \| \dot{\hat{x}}_{\text{root}}(t) - \dot{x}_{\text{root}}(t) \|^2 \right).
$$

#### 3.5 Domain Adaptation
To address the sim-to-real gap:
1. **Domain Randomization**: Randomize environmental factors (e.g., friction, mass).
2. **Latent Embedding Adjustment**: Refine latent dynamics $\mathbf{z}$ during real-robot deployment.

<figure>
  <div style="text-align:center">
    <img src="images/dog_rac_pipeline.png" alt="RAC Pipeline" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 2. Pipeline illustrating how RAC-generated data is processed and implemented for robotic control. The workflow includes generating animatable 3D models from monocular videos, retargeting the motion using inverse kinematics, and training control policies for deployment on quadruped robots.</figcaption>
</figure>

---

### 4. Results and Experiments

#### **Comparative Model Performance**

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>CD (cm) ↓</th>
      <th>F-score @ 2% ↑</th>
      <th>F-score @ 5% ↑</th>
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
    <img src="\fpost\rga\mimic_img\fig3.png" alt="Reconstruction to Robot Pipeline" style="width:90%;">
  </div>
  <figcaption style="text-align:center">
    Fig 3. The pipeline consists of three stages: The first row demonstrates the reconstruction of animal morphology and motion using RAC from real-world dog locomotion videos. The second row depicts the simulation of learned gaits in a virtual environment to refine policies. The third row illustrates the deployment of these policies on a physical quadruped robot, achieving realistic motion.
  </figcaption>
</figure>


---

### Conclusion

By integrating RAC's video-based 3D reconstruction with motion imitation, IK retargeting, and reinforcement learning, we enable quadruped robots to replicate lifelike animal gaits. This approach bypasses the need for intrusive motion capture setups by extracting natural motion data from videos, ensuring scalability and accessibility. Domain adaptation techniques ensure robust real-world deployment, demonstrating the versatility of data-driven robotic locomotion methods.