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
\- Attention map에서의 artifact를 정의하여 그 원인을 규명하고, 이러한 현상을 해석할 수 있는 가설 제시
\- Register token을 추가하여 ViT 아키텍처의 dense prediction task 성능 향상 (특히, DINOv2)

### Introduction

대량의 이미지를 활용해 사전 학습한 모델을 downstream task에 적용하는 것은 일반적인 접근법이다. 특히, [DINO](https://arxiv.org/abs/2104.14294)는 self-supervised로 학습하면서도 downstream task에서 준수한 성능을 보여주고, unsupervised segmentation도 가능하다는 점에서 주목받고 있다. 이를 바탕으로 DINO의 attention map을 활용한 object discovery 알고리즘인 [LOST](https://arxiv.org/abs/2109.14279)도 제안되었다.

[DINOv2](https://arxiv.org/abs/2304.07193)는 DINO를 확장하여 monocular depth estimation, semantic segmentation과 같은 dense prediction task에서 더욱 뛰어난 성능을 보였다. 하지만 **DINOv2가 LOST와 호환되지 않는 현상이 관찰되었고, 이는 DINOv2의 attention map에 존재하는 artifact 때문으로 추정**된다. 더 나아가, supervised ViT([DeiT](https://arxiv.org/abs/2012.12877), [OpenCLIP](https://arxiv.org/abs/2212.07143))에서도 유사한 artifact가 확인되었다. (Figure 2 참고)

<figure>
  <div style="text-align:center">
    <img src="/assets/img/vit_register/fig2.webp" alt="Fig 2" style="width:80%;">
  </div>
</figure>

Artifacts를 나타내는 **outlier**들은 다음의 특징을 가진다.
\- 약 10배 더 높은 norm을 가지며, 전체의 약 2%에 해당
\- 주로 middle layer에서 나타나며, 오래 학습하거나 모델이 큰 경우에 두드러짐
\- Local information을 버림
    \- 인근 patch와 유사도가 높아 original information (e.g., position, pixel)이 포함되지 않음
\- Global information을 포함
    \- Outlier patch에 classifier를 적용했을 때 일반 patch보다 높은 성능을 보여, 이미지의 global 정보를 담고 있음을 시사

이는 모델이 유용하지 않은 patch를 스스로 식별해 해당 spatial 정보를 버리고, global 정보를 효과적으로 표현하도록 학습한다는 것을 의미한다. ViT의 token 수가 제한된 상황에서 이러한 학습 과정은 global 정보를 최적화하려는 모델의 내재적 메커니즘으로 설명된다.

이를 해결하기 위해 register token을 추가했다.
\- Outlier token이 사라짐
\- Dense prediction task 성능이 향상
\- Feature map이 smooth해짐
    \- LOST를 활용한 object discovery 성능도 향상

### 2. Problem Formulation

#### 2.1 Artifacts in the Local Features of DINOv2

#### Artifacts are high-norm outlier tokens

![Figure 3](/assets/img/vit_register/fig3.webp){: .align-center}

- left: DINO와 DINOv2의 local feature norm을 시각화한 것으로, DINOv2에 존재하는 outlier가 high-norm인 것을 볼 수 있다.
- right: Small datasets에서 얻은 patch들의 분포를 나타내며, 임의로 지정한 cutoff value 150보다 높은 patch를 artifact로 정의한다. (모델에 따라 상이할 수 있음)

#### Outliers appear during the training of large models

![Figure 4](/assets/img/vit_register/fig4.webp){: .align-center}

- a: 전체 40 layer 중 15번째 layer부터 발견된다.
- b: 전체 training의 약 1/3 지점에서부터 발견된다.
- c: Large, Huge, Giant size에서 발견된다.

#### High-norm tokens appear where patch information is redundant

![Figure 5a](/assets/img/vit_register/fig5a.webp){: .align-center}

Artifact patch는 인접한 patch 4개와의 cosine 유사도가 높은 것을 보인다.

그렇다면 artifact patch는 어떤 정보를 갖고 있길래, 유사도가 높은 것일까?

#### High-norm tokens hold little local information

![Figure 5b](/assets/img/vit_register/fig5b.webp){: .align-center}

- Position prediction: 각 patch가 image 내에서 어디에 위치하는지를 예측 (positional embedding layer에서 position information이 주입되고, 이 정보가 얼마나 남아있는지를 예측)
- Pixel reconstruction

위 2개의 task에 대한 linear probing 성능이 낮기 때문에, local information이 artifact patch에 포함되어 있지 않다는 것을 알 수 있다.

#### Artifacts hold global information

![Table 1](/assets/img/vit_register/tab1.webp){: .align-center}

이번에는 image classification task에 대한 linear probing 성능이다. 여기서는 normal patch에 비해 outlier patch의 성능이 더 높다.

즉, outlier patch는 (normal patch에 비해) **local information 보다 global information을 더 포함하고 있으며, 이로 인해 인접한 patch와의 cosine similarity가 높다**고 볼 수 있다.

### 2.2 Hypothesis and Remediation

2.1 에서의 관측을 바탕으로 **충분히 학습된 큰 사이즈의 모델은 중복되는 token이 global information을 처리할 수 있게 한다**는 가설을 도출한다. 이러한 가설이 모델링 의도와는 일치하진 않지만 크게 문제되지는 않는다. 하지만 dense prediction task에서는 문제될 수 있다.

이를 해결하기 위해 register라는 additional token을 class token과 동일한 방식으로 추가한다. 그리고 register token은 inference에서 사용하지 않는다.
- 이러한 방식은 NLP 도메인의 Memory Transformer 논문에서 처음 적용되었다고 한다.
- 기존의 token들과 다른 점은 어떠한 정보도 주입되지 않고, token을 사용하지 않는다는 점이다.

물론 DINO에서는 왜 이러한 현상이 나타나지 않는지 규명하지 못 했다. 다만 DINO보다 모델 사이즈가 커지고, 학습 시간이 길어지면서 DINOv2에서 나타난 것으로 추정된다.

## <center> 3. Experiments

### 3.1 Training Algorithms and Data

- DeiT3: supervised (ImageNet-22k, ViT-B)
- OpenCLIP: text-supervised (Open source, ViT-B/16)
- DINOv2: self-supervised (ImageNet-22k, ViT-L)

### 3.2 Evaluation of the Proposed Solution

![Figure 7](/assets/img/vit_register/fig7.webp){: .align-center}

- Register token을 추가함으로써 patch norm이 크게 감소하는 것을 확인할 수 있다.

![Table 2](/assets/img/vit_register/tab2.webp){: .align-center}

- Segmentation, depth estimation 성능을 보면 dense prediction 성능이 향상된다.
- ImageNet 성능도 유지되거나 상승했다. (특히, DINOv2에서 0.5%p 상승)

![Figure 8](/assets/img/vit_register/fig8.webp){: .align-center}

- Top: register가 없는 경우 artifact가 나타난다.
- Bottom: register가 하나만 추가되더라도 dense prediction task 성능이 크게 향상된다.

### 3.3 Object Discovery

![Table 3](/assets/img/vit_register/tab3.webp){: .align-center}

VOC 2007 dataset에 대한 DINO + LOST의 성능이 61.9인데, DINOv2 + reg + LOST의 성능이 이에 미치지는 못한다. 그럼에도 register를 사용함으로써 상당한 성능 개선을 이룰 수 있다.

### 3.4 Qualitative Evaluation of Registers

![Figure 9](/assets/img/vit_register/fig9.webp){: .align-center}

흥미로운 점은 각 reg 토큰들이 각기 다른 object에 attention되어 있다는 것으로, 저자들은 이에 대한 future work를 제안한다.

## <center> Appendix

### A. Interpolation Artifacts and Outlier Position Distribution

![Figure 10](/assets/img/vit_register/fig10.webp){: .align-center}

![Figure 11](/assets/img/vit_register/fig11.webp){: .align-center}

Official DINOv2에서는 positional embedding이 16x16에서 7x7로 interpolate될 때 antialiasing을 사용하지 않았다. 그래서 Figure 11과 같은 gradient pattern을 갖게 되어, Figure 10 좌측 그래프처럼 outlier token이 vertical-striped pattern을 갖고 나타나게 된다.

반면 저자들은 antialiasing을 적용하여 Figure 10의 우측 그래프처럼 vertical pattern을 없앨 수 있었다. 이때 중심부보다는 가장자리에 outlier token이 많이 나타나는 것 또한, 대부분의 이미지가 object-centric하기에 가장자리 patch에 local information이 필요하지 않았을 것이라는 저자들의 가설을 뒷받침한다.
