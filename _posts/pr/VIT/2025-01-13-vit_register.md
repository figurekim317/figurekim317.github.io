---
layout: post
mathjax: true
image:  /assets/images/blog/post-5.jpg
title: "[Paper review] Vision transformer need registers"
last_modified_at: 2025-01-13
categories:
  - 논문리뷰
tags:
  - VIT
  - Computer Vision
  - AI
  - Meta
use_math: true
classes: wide
---

> ICLR 2024. [Paper](https://arxiv.org/abs/2309.16588) 
> Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski
> FAIR, Meta | Univ. Grenoble Alpes, Inria
> 11 Apr 2024  


<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig1.webp" alt="Fig 1" style="width:80%;">
  </div>
</figure>


### 0. Overview
ViT가 일부 patch를 global information 저장 용도로 활용하며 발생하는 **artifact** 현상을 분석하고 이를 극복할 수 있는 방안을 제시함
- **Artifact 정의 및 원인 분석**  
  - ViT가 low-informative 영역의 일부 패치를 재활용하여 global 정보를 저장  
  - 해당 패치의 local 정보가 손실되면서 모델 내부 연산(CLS 토큰 임베딩 등)에 활용됨  

- **Artifact 현상의 해석 및 가설 제시**  
  - Attention map에서 artifact가 형성되는 패턴 분석  
  - 모델이 inference 단계에서 자연스럽게 발생시키는 구조적 특징 탐구  

- **Dense prediction task 성능 향상 기법 제안**  
  - Register token을 추가하여 ViT의 global 정보 저장 방식 개선  
  - 특히, DINOv2에서의 성능 향상을 목표로 실험 및 검증

---

### 1. Introduction

대량의 이미지를 활용해 사전 학습한 모델을 downstream task에 적용하는 것은 일반적인 접근법이다. 특히, [DINO](https://arxiv.org/abs/2104.14294)는 self-supervised로 학습하면서도 downstream task에서 준수한 성능을 보여주고, unsupervised segmentation도 가능하다는 점에서 주목받고 있다. 이를 바탕으로 DINO의 attention map을 활용한 object discovery 알고리즘인 [LOST](https://arxiv.org/abs/2109.14279)도 제안되었다.

[DINOv2](https://arxiv.org/abs/2304.07193)는 DINO를 확장하여 monocular depth estimation, semantic segmentation과 같은 dense prediction task에서 더욱 뛰어난 성능을 보였다. 하지만 **DINOv2가 LOST와 호환되지 않는 현상이 관찰되었고, 이는 DINOv2의 attention map에 존재하는 artifact 때문으로 추정**된다. 더 나아가, supervised ViT([DeiT](https://arxiv.org/abs/2012.12877), [OpenCLIP](https://arxiv.org/abs/2212.07143))에서도 유사한 artifact가 확인되었다. (Figure 2 참고)

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig2.webp" alt="Fig 2" style="width:80%;">
  </div>
</figure>
<br>

Artifacts를 나타내는 **outlier**들은 다음의 특징을 가진다.
- 약 10배 더 높은 norm을 가지며, 전체의 약 2%에 해당
- 주로 middle layer에서 나타나며, 오래 학습하거나 모델이 큰 경우에 두드러짐
- Local information을 버림
    - 인근 patch와 유사도가 높아 original information (e.g., position, pixel)이 포함되지 않음
    - Image-level task에선 도움이 될 수 있으나 pixel-level task에선 불리하게 작용함
- Global information을 포함
    - Outlier patch에 classifier를 적용했을 때 일반 patch보다 높은 성능을 보여, 이미지의 global 정보를 담고 있음을 시사

이는 모델이 유용하지 않은 patch를 스스로 식별해 해당 spatial 정보를 버리고, global 정보를 효과적으로 표현하도록 학습한다는 것을 의미한다. ViT의 token 수가 제한된 상황에서 이러한 학습 과정은 global 정보를 최적화하려는 모델의 내재적 메커니즘으로 설명된다.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig6.png" alt="Fig 3" style="width:80%;">
  </div>
</figure>

이를 해결하기 위해 register token을 추가했다.
- ViT 기반의 다양한 방법론에 적용시, 다양한 task에서 기존의 성능을 유지하면서 더 정확한 feature map 및 attention map이 생성됨을 확인
  - Outlier token이 사라짐
  - Dense prediction task 성능이 향상
  - Feature map이 smooth해짐
    - LOST를 활용한 object discovery 성능도 향상

---

### 2. Problem Formulation

#### 2.1 Artifacts in the Local Features of DINOv2

##### **Artifacts are high-norm outlier tokens**
<br>
<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig3.webp" alt="Fig 3" style="width:80%;">
  </div>
</figure>
<br>
- **Left**: DINO와 DINOv2의 local feature norm을 시각화한 결과, DINOv2에서는 **outlier tokens**가 다른 token들보다 훨씬 높은 **high-norm** 값을 가지는 것이 확인됨
- **Right**: Small dataset에서 얻은 patch token의 norm 분포를 나타내며, 분포가 bimodal 형태를 띔  
  - Cutoff value를 150으로 설정해, 이를 초과하는 token을 **artifact**로 정의  
  - 이 값은 모델에 따라 다를 수 있지만, 이후 분석에서는 norm이 150을 초과하는 token을 "high-norm" 또는 "outlier"로 간주

<br>

##### **Outliers appear during the training of large models**

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig4.webp" alt="Fig 4" style="width:80%;">
  </div>
</figure>

- **a**: 40개의 layer 중 **15번째 layer**에서 norm의 크기 차이가 현저한 patch 등장
- **b**: ViT-g 모델에서 712k의 합습 iteration 중 **1/3 지점**부터 norm의 크기 차이가 현저한 patch 등장
- **c**: ViT-L보다 **사이즈가 큰 모델에서만** artifact가 등장함

<br>

##### **High-norm tokens appear where patch information is redundant**
<figure style="display: flex; justify-content: center; gap: 10px;">
  <div style="flex: 1; text-align: center;">
    <img src="/assets/img/vit_register/fig5a.webp" alt="Fig 5a" style="width: 70%;">
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/assets/img/vit_register/fig2.webp" alt="Fig 2" style="width: 120%;">
  </div>
</figure>

- 어떤 특징을 갖는 patch가 artifact로 재활용 되는가를 파악하기 위한 분석
- 이미지가 ViT의 encoder에 입력되기 직전에 수행되는 patch embedding layer값을 분석에 사용
- 한 patch의 embedding과 인근 4개의 patch embedding 사이의 유사도를 계산
- Artifact와 나머지 patch로 구분하여 유사도 분포를 시각화 (Fig 5a)
- 일반 patch에 비하여 artifact들은 인근 영역과 초기 patch embedding 유사도가 매우 크게 분포
- Artifact가 생성되는 영역은 인근 영역과 중복되는 특징을 갖고 있다고 유추 가능
  - **Redundant and low-informative**
- 정성적으로도 단조로운 색상과 특징을 갖는 배경에 artifact들이 주로 분포함을 통해 확인 가능

<br>

##### **High-norm tokens hold little local information**

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig5b.webp" alt="Fig 5b" style="width:60%;">
  </div>
</figure>

- Artifact가 local information을 상실했는지 분석하기 위해 두 가지 task에서 성능을 측정
- 두 task 수행을 위해 DINOv2 ViT-g 모델을 사용
1. **Position prediction**
  - Patch의 representation을 input, position을 label로 사용하여 해당 patch의 위치를 예측할 수 있도록 linear model을 지도 학습하여 artifact와 일반 patch 간의 prediction 성능을 비교
  - 결과: Artifact patch의 **accuracy 낮아**, 위치 정보(position information)를 거의 포함하지 않음

2. **Pixel reconstruction**
  - Patch의 representation을 input, pixel value를 정답으로 사용하여 해당 patch의 pixel value를 복원할 수 있도록 linear model을 지도학습하여 artifact와 일반 patch간의 reconstruction 성능을 비교
  - 결과: Artifact patch는 일반 patch보다 **reconstruction 성능 낮음**

이 결과는 **Artifact patch가 다른 일반 patch에 비하여 local information을 거의 포함하지 않는다는 사실**을 보여줌

<br>

##### **Artifacts hold global information**

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/tab1.webp" alt="Tab 1" style="width:80%;">
  </div>
</figure>

- Artifact가 global information를 포함하고 있는지 확인
- Image classification task에서 **linear probing**을 수행해, artifact patch가 global 정보를 포함하고 있는지 확인
  - Random으로 선택된 normal patch와 artifact patch 각각에 대해 logistic regression 모델을 학습하여 classification 성능을 비교
  - 결과: Artifact의 representation은 CLS token representation만큼 높은 Image Classification 성능을 보임

이는 Artifact patch가 **CLS token과 같이 풍부한 global information을 지니고 있다**고 유추할 수 있음

<br>

##### **Artifacts 특징 recap**
- Artifact는 큰 Vision Transformer 모델에서 학습 과정 중간에 등장함
- Artifact는 인근 영역 patch에 redundant한 특징을 가지고 있음
- Artifact는 해당 영역에 대한 local information을 거의 가지고 있지 않음
- Artifact는 CLS token만큼이나 많은 global information을 가지고 있음
- Artifact의 patch feature norm과 attention score가 일반 patch보다 매우 큼

---

#### 2.2 Hypothesis and Remediation

##### **Hypothesis**
- 충분히 큰 모델이 충분히 학습되면, 중복되는 artifact을 **global information**을 저장하고 처리하는 데 사용하도록 학습된다는 가설을 도출
- 이러한 현상이 자체적으로는 문제는 아니지만, dense prediction task에서는 **local information이 손실**되어 성능 저하를 초래할 수 있음

<br>

##### **Remediation**

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig6.png" alt="Fig 3" style="width:80%;">
  </div>
</figure>

이를 해결하기 위해 **register token**을 input sequence에 추가
1. **추가 위치**: Patch embedding layer 이후에 추가
2. **특징**: Learnable한 값으로 초기화되며, [CLS] token과 유사한 방식으로 동작
3. **사용 방식**:
    - Training 동안 모델이 register token을 사용해 global 정보를 처리
    - Inference 시 register token은 제거되고, [CLS] token과 patch token만 사용

---

### 3. Experiments

#### 3.1 Training Algorithms and Dataset  

##### **Backbone Model**  
- DeiT-III: Pretrained on ImageNet with supervised learning  
- OpenCLIP: Pretrained using text-supervised learning  
- DINOv2: Pretrained with self-supervised learning  

##### **Tasks and Methods**
- Image Classification  
  - Dataset: ImageNet  
  - Method: Linear probing  

- Image Segmentation  
  - Dataset: ADE20K 
  - Method: Segmentation with additional linear layer  

- Monocular Depth Estimation  
  - Dataset: NYUd  
  - Method: BinsFormer

---

#### 3.2 Evaluation of the Proposed Solution

##### **Register Tokens의 효과**
<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig7.webp" alt="Fig 7" style="width:80%;">
  </div>
</figure>

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/tab2.webp" alt="Tab 2" style="width:80%;">
  </div>
</figure>

- **Patch Norm 감소** 
  - Register token을 추가하면 **artifact가 제거**되며, patch norm이 안정화됨
  - Image-level task와 pixel-level task에서 모두 성능이 유지되거나 향상되는 결과를 확인함
  - 추가적으로 OpenCLIP을 이용한 zero-shot classification에서도 성능이 향상됨

<br>

##### **Register Token 개수와 성능**

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig8.webp" alt="Fig 8" style="width:80%;">
  </div>
</figure>
- Register token이 없는 경우 부터 16개인 경우까지 DINOv2 모델을 각각 학습하고 attention map의 상태변화와 downstream task성능의 변화를 분석
- Register token이 하나만 추가되어도 attention map의 artifact가 제거됨
- Image-level task인 image classificatio은 register token의 수가 증가될 수록 성능 향상
- Pixel-level task에서는 optimal register 개수가 있는 것으로 확인

---

#### 3.3 Object Discovery

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/tab3.webp" alt="Fig 3" style="width:80%;">
  </div>
</figure>

- DINOv2와 DeiT-III
  - Register Token 추가 시 object discovery 성능 대폭 향상  
- OpenCLIP
  - Register Token을 추가하면 성능이 약간 하락 
- DINOv2 VOC 2007 성능  
    - 기존 DINO : 61.9 corloc 
    - Register Token 없이 DINOv2: 35.3 corloc 
    - Register Token 추가 후 DINOv2: 55.4 corloc (+20.1 향상) 

결과적으로, DINOv2의 object discovery은 기존 DINO만큼은 도달하지 못했으나, Register Token을 통해 상당한 성능 향상을 확인  

---

#### 3.4 Qualitative Evaluation of Registers  

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig9.webp" alt="Fig 9" style="width:80%;">
  </div>
</figure>

- Register Token의 동작 방식을 정성적으로 분석  
- 모든 Register Token이 동일한 attention 패턴을 보이는지 확인  
- Register Token 간 차별화된 행동이 나타나는지 검증  
- Class Token 및 Register Token의 attention map을 시각화하여 패치 토큰과의 상호작용 분석  
- Register Token들은 완전히 동일한 행동을 보이지 않음  
- 일부 Register Token은 특정 객체에 집중하는 특이한 attention 패턴을 형성  
- 특정한 강제 없이도 자연스러운 attention 다양성이 발생  

**Register Token의 내부 구조와 regularization 기법을 추가적으로 연구할 필요성**을 시사

---

## Appendix

#### A. Interpolation Artifacts and Outlier Position Distribution

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig10.webp" alt="Fig 10" style="width:80%;">
  </div>
</figure>


<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig11.webp" alt="Fig 11" style="width:80%;">
  </div>
</figure>

- **DINOv2의 Positional Embedding Issue**
  - 16×16 positional embedding을 7×7로 interpolate할 때, **antialiasing 미적용**으로 gradient pattern이 생성(Figure 11).
  - 이로 인해 **vertical-striped outlier pattern**이 발생(Figure 10, 좌측 그래프)

- **해결 방안**
  - Antialiasing 적용 시 vertical pattern이 사라지고, outlier token이 중심부보다 가장자리에 주로 나타남(Figure 10, 우측 그래프)
  - 이는 대부분의 이미지가 object-centric하기 때문에, 가장자리 patch에서 **local information 필요성이 적음**을 뒷받침