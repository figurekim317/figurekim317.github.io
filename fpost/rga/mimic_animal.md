---
layout: fpost
title: "Project 8"
permalink: /fpost/rga_rl/
author: Dohyeong Kim
tags:   
  - Autonomy driving
  - Object detection
  - Segmentation
  - Model Compression
  - Optimization
---

### • Goal
- Combine ideas from **“Learning Agile Robotic Locomotion Skills by Imitating Animals”** with **“RAC: Reconstructing Animatable Categories from Videos”**  
- Develop a pipeline that converts **YouTube or monocular videos** of animals (e.g., dogs) into **3D skeletal motion**, retargets the motion to a **robot**, and trains it via **reinforcement learning**  
- Verify that **domain adaptation** strategies enable **real-time, real-world** deployment of such motion on physical quadruped robots

<br/>

---

## 1. Problem Definition & Dataset Analysis

- **Context**  
  - Traditionally, motion capture (mocap) data from real animals (e.g., dogs) is used to learn agile and dynamic locomotion skills. However, mocap often requires specialized equipment and setups.  
  - **RAC** enables per-video, **3D reconstructions** of animals captured in casual, in-the-wild videos (e.g., on YouTube).  
  - By combining **RAC**’s ability to reconstruct animatable 3D models with the **motion imitation** approach from [Peng et al.], we aim to replicate lifelike animal gaits on a **quadruped robot**.

- **Key Observations**  
  1. **Between-Instance Variation**: Different dog breeds (or animals) exhibit diverse body proportions (limb lengths, ear shapes, etc.).  
  2. **Within-Instance Variation**: Each dog’s motion over time includes skeletal articulation and soft deformation (e.g., muscles, fur).  
  3. **Sim-to-Real Gap**: Policies trained in simulation often fail on real hardware without **domain adaptation** due to unmodeled dynamics (friction, motor torques, etc.).

<figure>
  <div style="text-align:center">
    <img src="images/dog_motion_example.png" alt="Dog Locomotion Example" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 1. Example of a dog’s gait extracted from a casual YouTube video (conceptual).</figcaption>
</figure>

<br/>

---

## 2. Proposed Model & Approach

### • Using **RAC** for 3D Reconstruction
- **RAC (Reconstructing Animatable Categories)**:  
  - Learns a **category-level** skeleton (e.g., for dogs) with a **morphology code** $ \beta $ per instance/video.  
  - Decomposes motion into **articulation** (joint rotations) and **soft deformation** (non-rigid warping).  
  - Incorporates a **background model** (NeRF) for robust rendering and better silhouette refinement.

### • From Video to Robot
1. **Video Input**: Collect single-view or multi-view videos of dog locomotion.  
2. **RAC Reconstruction**: Obtain a **3D canonical model** + per-frame articulations $ \theta $ + morphological differences ($ \Delta J_\beta $).  
3. **Retargeting**: Map the resulting 3D joint trajectories to the **robot** via **Inverse Kinematics** (IK).  
4. **Motion Imitation**: Use reinforcement learning (RL) to train the robot’s policy $ \pi_\theta $ in simulation, imitating the retargeted reference motions.  
5. **Domain Adaptation**: Transfer the learned policy to the physical quadruped robot, mitigating the sim-to-real gap.

<figure>
  <div style="text-align:center">
    <img src="images/dog_rac_pipeline.png" alt="RAC Pipeline" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 2. Simplified pipeline: (1) Videos → (2) RAC reconstruction → (3) IK retargeting → (4) RL-based motion imitation → (5) Domain adaptation and real-world deployment.</figcaption>
</figure>

<br/>

---

## 3. Implementation & Training

### • Step-by-Step Process

1. **RAC Reconstruction**  
   - **Between-Instance** Variation: A morphology code $ \beta $ adjusts bone length, shape, and appearance.  
   - **Within-Instance** Variation: Per-frame articulation $ \theta $ and invertible soft-deformation fields.  
   - **Differentiable Rendering**: Uses silhouettes, RGB, and optical flow to optimize the 3D model and background NeRF end-to-end.

2. **Motion Retargeting**  
   - After we obtain time-varying joint positions $ \hat{x}_i(t) $ from the RAC output, we solve the IK problem to match them to the robot’s joint variables $ q_t $.  
   - Formally ([Peng et al.], Eq. (1)):

   $$
   \min_{q_{0:T}}
   \sum_{t} \sum_{i}
   \|\hat{x}_i(t) - x_i(q_t)\|^2
   + (\bar{q} - q_t)^T W (\bar{q} - q_t).
   $$

3. **Motion Imitation (RL)**  
   - In simulation (e.g., PyBullet, Mujoco), define a reward function that measures how closely the robot tracks the reference joint angles, velocities, and end-effector trajectories.  
   - Example reward ([Peng et al.], Eqs. (4)–(9)) could be:

   $$
   r_t = w_p \, r_t^p + w_v \, r_t^v + w_e \, r_t^e + w_{rp} \, r_t^{rp} + w_{rv} \, r_t^{rv},
   $$

   where $ r_t^p $ focuses on pose accuracy, $ r_t^v $ on velocity matching, etc.

4. **Domain Adaptation**  
   - **Domain Randomization**: Randomize friction, mass, motor parameters during training.  
   - **Latent Embedding** ($ \mathbf{z} $): Learned representation of environment dynamics that can be adjusted for real hardware.  
   - During real-robot trials, refine $ \mathbf{z} $ or the policy to handle physical discrepancies (motor torque limits, real friction, sensor noise).

<br/>

### • Training Settings

- **Simulation**: Typically trained with tens or hundreds of millions of timesteps using PPO or SAC.  
- **Hardware**: The final policy is deployed on a **quadruped robot** (e.g., Unitree, MIT mini-cheetah, or similar).  
- **Time Horizons**: Usually 5–10 seconds per episode for locomotion tasks.

<br/>

---

## 4. Key Challenges & Solutions

1. **Unstable Single-View Reconstruction**  
   - **Challenge**: Monocular videos can cause ambiguities in 3D shape or skeleton inference.  
   - **Solution**: Use additional priors (category skeleton, shape regularization) or, if possible, multi-view data to improve reliability.

2. **Overly Complex Deformation**  
   - **Challenge**: Overfitting can occur if the soft deformation field tries to “explain everything.”  
   - **Solution**: Regularize the bone-based articulation vs. soft deformation boundaries, ensuring stable shape and motion.

3. **Sim-to-Real Gap**  
   - **Challenge**: Policies that work in simulation might fail when friction, sensor noise, or motor torque differ in reality.  
   - **Solution**: Domain randomization + policy adaptation. For instance, searching for an optimal latent vector $ \mathbf{z}^* $ that maximizes performance on the real robot.

4. **Real-Time Control**  
   - **Challenge**: High-dimensional policies or large neural nets might be slow to run on embedded hardware.  
   - **Solution**: Optimize network size, use TensorRT or similar acceleration, or offload to a compact controller.

<br/>

---

## 5. Potential Extensions & Future Directions

1. **Complex Motions**  
   - Expand beyond straightforward walking/trotting to include **jumping**, **obstacle avoidance**, or **spinning** behaviors.  
   - Gather additional YouTube videos capturing more dynamic dog or cat movements.

2. **Multi-Camera or Improved 3D Keypoint Systems**  
   - If single-view reconstructions remain noisy, consider multi-camera setups or advanced pose-estimation techniques to refine 3D data quality.

3. **Online Adaptation**  
   - Continually update the policy on the real robot using real-time feedback (IMU, foot contacts) for improved robustness and fast domain adaptation.

4. **Safety & Energy Efficiency**  
   - Integrate constraints to reduce risk of falls or hardware damage.  
   - Investigate gait patterns that minimize energy consumption or motor heat.

<figure>
  <div style="text-align:center">
    <img src="images/real_robot_deployment.png" alt="Robot Deployment" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 3. Conceptual depiction of a quadruped robot performing dog-like gaits extracted from YouTube footage.</figcaption>
</figure>

<br/>

---

## 6. Results & Conclusion

- **Enhanced Motion Quality**  
  - Combines **RAC** (which captures realistic animal shapes and articulations) with **motion imitation** RL to achieve **lifelike gaits** on quadruped robots.  

- **Robustness via Domain Adaptation**  
  - Policies become resilient to real-world discrepancies (friction, sensor noise) thanks to domain randomization and latent embedding adjustments.

- **Scalable Data Source**  
  - Bypasses specialized mocap setups by leveraging **YouTube** or casually captured videos, greatly expanding the variety of reference motions.

- **Real-Time Possibility**  
  - With optimized model sizes and efficient inference frameworks, near real-time control (tens to hundreds of Hz) is feasible on modern robotic platforms.

<br/>

**In summary**, by integrating **RAC**’s video-based 3D reconstruction with the motion imitation pipeline from [Peng et al.]—including retargeting, RL training, and domain adaptation—we can empower quadruped robots to learn agile, animal-like behaviors purely from ordinary videos. This paves the way for more flexible, data-driven robotic locomotion, unbound by heavy motion capture equipment or specialized lab environments.
