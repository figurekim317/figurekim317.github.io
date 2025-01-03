---
layout: fpost
title: "Project 2"
permalink: /fpost/snu/snn/
#date:   2017-05-26 15:05:55 +0300
image:  /assets/images/blog/post-5.jpg
author: Dohyeong Kim
tags:  
    - Spiking neural networks
    - Object detection
    - Model Compression
    - Optimization
---


### • Goal

#### \- Improve DNN to SNN mapping algorithm accuracy

  - \- Development of multi-object detection algorithm based on CNN
  - \- SNN based multi object detection algorithm optimization
  - \- Algorithm lightweighting and optimization for autonomous driving for vehicles
  - \- Algorithm lightweighting and accuracy improvement for SNN implantation
  - \- Improving DNN to SNN Mapping Algorithm Performance by Considering Inference in SNN
  - \- SNN error rate compared to DNN achieved < 5%

<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf2/snn conversion scheme.PNG" alt="대체 텍스트" style="width:70%;">
  </div>
  <figcaption style="text-align:center">Fig 1. The CNN-SNN conversion scheme.</figcaption>
</figure>

<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf2/channel-wise normalization.PNG" alt="대체 텍스트" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Fig 2. (a) The image shows the normalized maximum activation values ​​obtained through channel-specific normalization in the eight convolutional layers of the TinyYOLO model. The blue and red lines represent the mean and minimum of the normalized activation values, respectively. (b) The proposed channel-by-channel normalization is a normalization method that targets $A^l_j$, which is all activation matrices (i.e., feature maps) of the $l$-th layer.</figcaption>
</figure>


<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf2/yoloresult.PNG" alt="대체 텍스트" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Fig 3. Object detection results (TinyYOLO vs Spiking-YOLO with layer-norm vs Spiking-YOLO with channel-norm).</figcaption>
</figure>


<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf2/accuracy latency tradeoff.png" alt="대체 텍스트" style="width:50%;">
  </div>
  <figcaption style="text-align:center">Fig 4. Accuracy latency trade-off.</figcaption>
</figure>

<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf2/bayesain opt.png" alt="대체 텍스트" style="width:75%;">
  </div>
  <figcaption style="text-align:center">Fig 5. Bayesian threshold optimization overview.</figcaption>
</figure>

<figure>
  <div style="text-align:center">
    <img src="/fpost/images/pf2/bayesian2.png" alt="대체 텍스트" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Fig 6. (a) Two-phase threshold voltages for fast and accurate object detection in SNNs. (b) Object detection accuracy (mAP %) curve of phase-1 threshold voltages as time step increases.</figcaption>
</figure>



#### \- HW/SW Integration

- \- Implementation and optimization of the proposed algorithm on the chip using the provided API
- \- Demonstration of SNN-based multi-object recognition algorithm on digital emulator, implementation of core functions such as ReLU and Pooling, implementation of parallel data interface