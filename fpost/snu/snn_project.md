---
layout: fpost
title: "Project 3"
permalink: /fpost/snu/snn_project/
#date:   2017-05-26 15:05:55 +0300
image:  /assets/images/blog/post-5.jpg
author: Jongwan Kim
tags:   
   - Spiking neural network
   - Object detection
   - Optimization
   - Noise robust model
---

### • Goal

#### \- Designing an Efficient SNN Learning Algorithm for Vertical Synapse Array Structure
   - \- Suggested SNN learning method using knowledge distillation for resource-constrained environment of 3D-CIM
   - \- Implementation of an algorithm to reduce the occurrence of spikes for energy efficiency in model inference
   - \- Compute capability sensitivity analysis


#### \- Implementation of an artificial neural network inference model that is robust to device noise
   - \- SNN model design robust to time-variant noise generated in 3D CIM
   - \- To address the noise from high-dimensional data and parameters in artificial neural network training, robust optimization techniques are used
   - \- By obtaining observed values that consider the worst-case scenario along with the original values, this approach aims to overcome the limitations of ECOC in neural network training
   - \- Analyze and apply model ensemble technology, which is a type of Bayesian model

<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf3/summary.png" alt="대체 텍스트" style="width:50%;">
  </div>
  <figcaption style="text-align:center">Fig 1. System diagram of task performance.</figcaption>
</figure>

<figure>
  <div style="text-align:center">
    <div style="display:inline-block; margin-right:1px;">
      <img src="/fpost/images/pf3/BNN1.png" alt="대체 텍스트" style="width:70%;">
    </div>
    <div style="display:inline-block;">
      <img src="/fpost/images/pf3/BNN2.png" alt="대체 텍스트" style="width:70%;">
    </div>
  </div>
    <figcaption style="text-align:center">Fig 2. BNN implementation method with -1/1 as activation variable.</figcaption>
</figure>


<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf3/BWN.png" alt="대체 텍스트" style="width:50%;">
  </div>
  <figcaption style="text-align:center">Fig 3. BWN-based CIM circuit configuration</figcaption>
</figure>

<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf3/noise robus.png" alt="대체 텍스트" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 4. Comparison of the accuracy and number of spikes of various neural coding methods when spike deletion (left) and spike jitter (right) noise were applied respectively (VGG16 model, CIFAR-10 dataset).</figcaption>
</figure>

<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf3/noise snn.png" alt="대체 텍스트" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 5. Accuracy comparison of various neural coding and proposed weight scaling methods when spike deletion (left) and spike jitter (right) noise are applied respectively (VGG16 model, CIFAR-10 dataset).</figcaption>
</figure>

