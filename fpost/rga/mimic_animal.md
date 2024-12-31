---
layout: fpost
title: "Learning Agile Locomotion by Imitating Animals for Quadruped Robots"
permalink: /fpost/rga/mimic_animal/
author: Dohyeong Kim
tags:   
  - Animal Locomotion
  - RAC
  - Reinforcement Learning
  - Motion Retargeting
  - Domain Adaptation
  - 3D Reconstruction
  - Robotics
  - Sim-to-Real Transfer
---

## Goal
- Combine ideas from **"Learning Agile Robotic Locomotion Skills by Imitating Animals"** and **"RAC: Reconstructing Animatable Categories from Videos"**.
- Develop a pipeline to extract **3D skeletal motion** from monocular or YouTube videos of animals and retarget it to a **quadruped robot** for training via **reinforcement learning (RL)**.
- Achieve **real-time deployment** on physical robots by addressing sim-to-real challenges through **domain adaptation**.

---

### 1. Problem Definition & Dataset Analysis

#### **Context**
- Traditional approaches rely on specialized **motion capture (mocap)** setups to record animal locomotion, which are resource-intensive.
- **RAC** enables **3D reconstruction** of animals directly from casual, in-the-wild videos.
- This project aims to combine RAC's capabilities with RL-based motion imitation to replicate naturalistic animal movements on robots, such as dogs’ gaits.

#### **Key Observations**
1. **Between-Instance Variation**: Diverse animal breeds differ significantly in morphology (e.g., limb length, tail movement).
2. **Within-Instance Variation**: Individual movements vary dynamically over time (e.g., joint articulation, fur deformation).
3. **Sim-to-Real Gap**: Discrepancies in simulation and real-world conditions (e.g., motor torques, friction) often hinder direct deployment.

<figure>
  <div style="text-align:center">
    <img src="images/dog_motion_example.png" alt="Dog Locomotion Example" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 1. Example of a dog’s gait extracted from a casual YouTube video.</figcaption>
</figure>

---

### 2. Proposed Model & Approach

#### **Using RAC for 3D Reconstruction**
- **RAC (Reconstructing Animatable Categories)**:
  - Learns a category-level skeletal structure with a **morphology code** $ \beta $ for each instance.
  - Decomposes motion into:
    - **Articulations** $ \theta $: Joint rotations.
    - **Soft Deformations** $ \Delta J_{\beta} $: Non-rigid body warping.
  - Utilizes NeRF for background rendering and improved silhouette refinement.

#### **Pipeline**
1. **Video Input**: Collect single-view or multi-view locomotion videos.
2. **RAC Reconstruction**:
   - Generate a canonical 3D model, per-frame articulations $ \theta $, and morphology differences $ \Delta J_{\beta} $.
3. **Motion Retargeting**:
   - Map the extracted joint trajectories to a quadruped robot using **Inverse Kinematics (IK)**:
   $$ \min_{q_{0:T}} \sum_t \sum_i \| \hat{x}_i(t) - x_i(q_t) \|^2 + (\bar{q} - q_t)^T W (\bar{q} - q_t). $$
4. **Motion Imitation (RL)**:
   - Train a policy $ \pi_\theta $ in simulation to mimic reference motions.
5. **Domain Adaptation**:
   - Transfer learned policies to the real robot while mitigating dynamics mismatches.

<figure>
  <div style="text-align:center">
    <img src="images/dog_rac_pipeline.png" alt="RAC Pipeline" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 2. Pipeline: From videos to RAC reconstruction, retargeting, RL imitation, and domain adaptation.</figcaption>
</figure>

---

### 3. Implementation & Training

#### **Step-by-Step Process**

1. **RAC Reconstruction**:
   - Capture instance-level differences via $ \beta $.
   - Optimize articulation $ \theta $ and soft deformations $ \Delta J_{\beta} $ for accurate 3D modeling.

2. **Motion Retargeting**:
   - Map time-varying joint positions $ \hat{x}_i(t) $ to robot joint angles $ q_t $ using the IK optimization problem.

3. **Motion Imitation (RL)**:
   - Define a reward function $ r_t $ to incentivize close matching of reference poses:
     $$ r_t = w_p r_t^p + w_v r_t^v + w_e r_t^e + w_{rp} r_t^{rp} + w_{rv} r_t^{rv}, $$
     where $ r_t^p $ measures pose accuracy, $ r_t^v $ measures velocity consistency, etc.

4. **Domain Adaptation**:
   - Apply **domain randomization** to bridge the sim-to-real gap by varying parameters (e.g., friction, mass):
     $$ \mathbf{z}^* = \arg\max_{\mathbf{z}} J(\pi_\theta, \mathbf{z}). $$

#### **Training Settings**
- **Simulation**: Use PyBullet or MuJoCo for efficient training over millions of steps.
- **Hardware**: Deploy optimized policies on Unitree or MIT mini-cheetah robots.

---

### 4. Challenges & Solutions

#### 1. **Unstable Reconstruction**
- **Challenge**: Single-view inputs introduce ambiguities.
- **Solution**: Incorporate priors or multi-view data to enhance stability.

#### 2. **Overfitting Deformations**
- **Challenge**: Excessive soft deformation modeling can cause overfitting.
- **Solution**: Regularize deformation parameters to balance rigidity and flexibility.

#### 3. **Sim-to-Real Gap**
- **Challenge**: Simulation-trained policies may fail in real environments.
- **Solution**: Employ domain randomization and train latent embeddings to adapt dynamics.

#### 4. **Real-Time Constraints**
- **Challenge**: Policies may be computationally intensive.
- **Solution**: Optimize for low-latency inference using TensorRT.

---

### 5. Results & Conclusion

- **Enhanced Locomotion Quality**:
  - Integrates RAC for lifelike 3D motion capture and RL for efficient policy learning.
- **Robust Deployment**:
  - Combines domain randomization and adaptation for real-world robustness.
- **Scalable Data**:
  - Exploits casual video sources (e.g., YouTube) to build diverse locomotion datasets.
- **Real-Time Capability**:
  - Ensures fast, responsive control suitable for embedded robotic systems.

---

### References
1. Peng et al., *Learning Agile Robotic Locomotion Skills by Imitating Animals*.
2. RAC: *Reconstructing Animatable Categories from Videos*.
3. NVIDIA TensorRT Documentation.
4. MuJoCo and PyBullet for RL Training.
