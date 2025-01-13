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

> ICLR 2024. [[Paper](https://arxiv.org/abs/2309.16588)]  
> Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski
> FAIR, Meta | Univ. Grenoble Alpes, Inria
> 11 Apr 2024  


<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig1.png" alt="Fig 1" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Fig 1. Register tokens enable interpretable attention maps in all vision transformers, similar to the original DINO method</figcaption>
</figure>

### Abstract
- Attention map에서의 artifact를 정의하여 그 원인을 규명하고, 이러한 현상을 해석할 수 있는 가설 제시
- Register token을 추가하여 ViT 아키텍처의 dense prediction task 성능 향상 (특히, DINOv2)

### Introduction

대량의 이미지를 활용해 사전 학습한 모델을 downstream task에 적용하는 것은 일반적인 접근법이다. 특히, [DINO](https://arxiv.org/abs/2104.14294)는 self-supervised로 학습하면서도 downstream task에서 준수한 성능을 보여주고, unsupervised segmentation도 가능하다는 점에서 주목받고 있다. 이를 바탕으로 DINO의 attention map을 활용한 object discovery 알고리즘인 [LOST](https://arxiv.org/abs/2109.14279)도 제안되었다.

[DINOv2](https://arxiv.org/abs/2304.07193)는 DINO를 확장하여 monocular depth estimation, semantic segmentation과 같은 dense prediction task에서 더욱 뛰어난 성능을 보였다. 하지만 **DINOv2가 LOST와 호환되지 않는 현상이 관찰되었고, 이는 DINOv2의 attention map에 존재하는 artifact 때문으로 추정**된다. 더 나아가, supervised ViT([DeiT](https://arxiv.org/abs/2012.12877), [OpenCLIP](https://arxiv.org/abs/2212.07143))에서도 유사한 artifact가 확인되었다. (Figure 2 참고)

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig2.webp" alt="Fig 2" style="width:80%;">
  </div>
</figure>

Artifacts를 나타내는 **outlier**들은 다음의 특징을 가진다.
- 약 10배 더 높은 norm을 가지며, 전체의 약 2%에 해당
- 주로 middle layer에서 나타나며, 오래 학습하거나 모델이 큰 경우에 두드러짐
- Local information을 버림
    - 인근 patch와 유사도가 높아 original information (e.g., position, pixel)이 포함되지 않음
- Global information을 포함
    - Outlier patch에 classifier를 적용했을 때 일반 patch보다 높은 성능을 보여, 이미지의 global 정보를 담고 있음을 시사

이는 모델이 유용하지 않은 patch를 스스로 식별해 해당 spatial 정보를 버리고, global 정보를 효과적으로 표현하도록 학습한다는 것을 의미한다. ViT의 token 수가 제한된 상황에서 이러한 학습 과정은 global 정보를 최적화하려는 모델의 내재적 메커니즘으로 설명된다.

이를 해결하기 위해 register token을 추가했다.
- Outlier token이 사라짐
- Dense prediction task 성능이 향상
- Feature map이 smooth해짐
    - LOST를 활용한 object discovery 성능도 향상

### 2. Problem Formulation

#### 2.1 Artifacts in the Local Features of DINOv2

##### Artifacts are high-norm outlier tokens

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig3.webp" alt="Fig 3" style="width:80%;">
  </div>
</figure>

- **Left**: DINO와 DINOv2의 local feature norm을 시각화한 결과, DINOv2에서는 **outlier tokens**가 다른 token들보다 훨씬 높은 **high-norm** 값을 가지는 것이 확인됨.
- **Right**: Small dataset에서 얻은 patch token의 norm 분포를 나타내며, 분포가 bimodal 형태를 띔.  
  - **Cutoff value**를 150으로 설정해, 이를 초과하는 token을 **artifact**로 정의.  
  - 이 값은 모델에 따라 다를 수 있지만, 이후 분석에서는 norm이 150을 초과하는 token을 "high-norm" 또는 "outlier"로 간주.


##### Outliers appear during the training of large models

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig4.webp" alt="Fig 4" style="width:80%;">
  </div>
</figure>

- **a**: 40개의 layer 중 **15번째 layer**에서 outlier token이 다른 token과 차별화되기 시작함.
- **b**: 모델 학습의 약 **1/3 지점**부터 outlier token이 나타남.
- **c**: 모델 크기에 따른 분석 결과, outlier는 **Large**, **Huge**, **Giant** 크기의 모델에서만 관찰됨.


#### Outlier Tokens의 Local Information Analysis

- Outlier tokens는 **local 정보가 부족**하다는 특징을 가짐:
  - Neighbor patches와의 **cosine similarity**를 분석한 결과, outlier tokens의 유사도가 normal tokens보다 낮음.  
  - **Position prediction** 및 **input patch reconstruction**을 통해 local information 보유량을 분석한 결과, outlier tokens는 normal tokens보다 낮은 성능을 보임. (Table 1)
- 결론적으로, outlier tokens는 local 정보를 희생하면서도 **global 정보를 더 효과적으로 담기 위한 모델의 학습 전략**을 반영한 것으로 보임.


#### High-norm tokens appear where patch information is redundant
<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig5a.webp" alt="Fig 5a" style="width:80%;">
  </div>
</figure>

- **Artifact patch**는 인접한 4개의 patch와 cosine 유사도가 높은 것으로 나타남.
- 이는 **redundant information**을 포함하고 있음을 의미하며, 모델이 이러한 정보를 제거해도 이미지 표현 품질에 큰 영향을 미치지 않는다는 것을 시사.
- Fig. 2에서 보이듯, 이런 patch는 종종 **uniform한 배경 영역**에서 발생.


#### High-norm tokens hold little local information

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig5b.webp" alt="Fig 5b" style="width:80%;">
  </div>
</figure>

**Artifact patch**의 local information을 분석하기 위해 두 가지 task에서 성능을 측정:
1. **Position prediction**:
    - Patch가 이미지 내에서 위치하는 좌표를 예측.
    - 결과: Artifact patch의 **정확도가 낮아**, 위치 정보(position information)를 거의 포함하지 않음.
2. **Pixel reconstruction**:
    - Patch로부터 원래의 픽셀 값을 복원.
    - 결과: Artifact patch는 일반 patch보다 **복원 정확도 낮음**.

이 결과는 **Artifact patch가 local 정보를 거의 포함하지 않는다는 사실**을 보여줌.


#### Artifacts hold global information

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/tab1.webp" alt="Tab 1" style="width:80%;">
  </div>
</figure>

- Image classification task에서 **linear probing**을 수행해, artifact patch가 global 정보를 포함하고 있는지 확인:
    - Random으로 선택된 normal patch와 artifact patch 각각에 대해 logistic regression 모델을 학습하여 분류 정확도를 비교.
    - 결과: Artifact patch의 정확도가 normal patch보다 **훨씬 높음**.

이는 Artifact patch가 **local information 대신 global information을 더 많이 포함**하고 있음을 의미함.


### 2.2 Hypothesis and Remediation

#### Hypothesis
- 충분히 큰 모델이 충분히 학습되면, 중복되는 patch token을 **global information**을 저장하고 처리하는 데 사용하도록 학습된다는 가설을 도출.
- 이러한 현상이 자체적으로는 문제는 아니지만, dense prediction task에서는 **local information이 손실**되어 성능 저하를 초래할 수 있음.

#### Remediation
이를 해결하기 위해 **register token**을 추가:
1. **추가 위치**: Patch embedding layer 이후에 추가.
2. **특징**: Learnable한 값으로 초기화되며, **[CLS] token**과 유사한 방식으로 동작.
3. **사용 방식**:
    - Training 동안 모델이 register token을 사용해 global 정보를 처리.
    - Inference 시 register token은 제거되고, [CLS] token과 patch token만 사용.

이 방식은 **NLP의 Memory Transformers**에서 처음 제안되었으며, vision transformer의 **interpretability 및 dense prediction task 성능** 문제를 해결하는 데 기여

---

#### 추가 관찰
- Artifact 현상은 모델 크기와 학습 길이에 따라 크게 좌우됨(Fig. 4 참고).
- Pretraining 방식 또한 영향을 미침: OpenCLIP 및 DeiT-III에서는 작은 모델 크기(B)와 큰 모델 크기(L)에서도 outliers가 관찰됨(Fig. 2 참고).
- 하지만 Artifact 현상이 왜 DINO에서는 나타나지 않는지는 완전히 규명되지 않음.  


### 3. Experiments

#### 3.1 Training Algorithms and Data

- **DeiT-III**: Supervised training on ImageNet-22k with ViT-B architecture.
- **OpenCLIP**: Text-supervised training using ViT-B/16 on a licensed text-image dataset.
- **DINOv2**: Self-supervised training on ImageNet-22k with ViT-L configuration.

---

#### 3.2 Evaluation of the Proposed Solution

##### Register Tokens의 효과
- **Patch Norm 감소**: Register token을 추가하면 **artifact가 제거**되며, patch norm이 안정화됨.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig7.webp" alt="Fig 7" style="width:80%;">
  </div>
</figure>

##### Downstream Task 성능
- Dense prediction task(예: Segmentation, Depth Estimation) 성능이 개선됨.
- **ImageNet Classification**에서도 성능 유지 또는 향상(DINOv2: +0.5%p).

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/tab2.webp" alt="Tab 2" style="width:80%;">
  </div>
</figure>

##### Register Token 개수와 성능
- Register token이 하나만 추가되어도 artifact가 제거되고, dense prediction task 성능이 크게 향상됨.
- **Optimal Register Token 수**: Dense prediction task에서는 최적의 register 수가 존재하며, ImageNet 성능은 register 수가 많아질수록 증가.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig8.webp" alt="Fig 8" style="width:80%;">
  </div>
</figure>


#### 3.3 Object Discovery

- **DINO + LOST 성능**: VOC 2007 기준, 61.9.
- **DINOv2 + Reg + LOST 성능**: 기존 DINOv2의 35.3에서 **20.1-point 향상**된 55.4를 기록.
- **OpenCLIP의 경우**: 일부 성능 감소가 관찰되었으나, 분석은 추가 연구로 제안.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/tab3.webp" alt="Fig 3" style="width:80%;">
  </div>
</figure>


#### 3.4 Qualitative Evaluation of Registers

- **Reg Token의 Attention Behavior**:
  - 일부 register token은 다양한 object에 attention을 할당하며, 자연스럽게 **다양한 패턴**을 형성.
  - 이는 명시적으로 설계되지 않은 자연 발생적인 현상.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig9.webp" alt="Fig 9" style="width:80%;">
  </div>
</figure>

- **Future Work**:
  - Reg token의 regularization 기법 및 추가 분석.


## Appendix

### A. Interpolation Artifacts and Outlier Position Distribution

- **DINOv2의 Positional Embedding Issue**:
  - 16×16 positional embedding을 7×7로 interpolate할 때, **antialiasing 미적용**으로 gradient pattern이 생성(Figure 11).
  - 이로 인해 **vertical-striped outlier pattern**이 발생(Figure 10, 좌측 그래프).

- **해결 방안**:
  - Antialiasing 적용 시 vertical pattern이 사라지고, outlier token이 중심부보다 가장자리에 주로 나타남(Figure 10, 우측 그래프).
  - 이는 대부분의 이미지가 object-centric하기 때문에, 가장자리 patch에서 **local information 필요성이 적음**을 뒷받침.

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