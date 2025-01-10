---
category: pr
layout: post
mathjax: true
image:  /assets/images/blog/post-5.jpg
title: "[Paper review] Contrastive Representation Learning: A Framework and Review"
last_modified_at: 2025-01-10
tags:
  - Contrastive Learning
  - Self-supervised learning
excerpt: "ReStyle Paper review"
use_math: true
classes: wide
---

> Contrastive Representation Learning: A Framework and Review [[Paper](https://ieeexplore.ieee.org/abstract/document/9226466)]  
> Phuc H. Le-Khac, Graham Healy, Alan F. Smeaton 
> ML-Labs, Dublin City University 
> 10 Oct 2020  

---

---
Interview를 보았는데 Contrastive learning에 대한 질문을 받았다. SimCLR로 간접적으로 설명하긴 했는데 내가 생각해도 형편없는 답변을 했다. 대학원 시절 랩실에서 가장 먼저 읽어서 랩세미나 시간에 발표하기도 했는데 너무 당연하다고 생각해오고 모델을 가져다가 쓰기만 하다보니 정작 말로 설명을 못하는 느낌이 들어 이번 주는 Contrastive learning 분야를 파면서 대표 논문 review를 진행하려고 한다. 
Facenet에서 triplet loss를 접하고 흥미 있는 분야로 생각했는데 self-supervised learning 분야에서 많이 발전을 이룬 것 같다. 

---

## Introduction
Contrastive Learning(CRL)이란 입력 샘플 간의 **비교**를 통해 학습을 하는 것이다. 
CRL의 경우에는 self-supervised learning에 사용되는 접근법 중 하나(물론 supervised learning의 맥락에서 CRL이 수행되기도 한다)로 사전에 정답 데이터를 구축하지 않는 판별 모델이라고 할 수 있다.

따라서, **데이터 구축 비용이 들지 않음**과 동시에 **학습 과정에 있어서 보다 용이한 장점**을 갖는다. 이러한 데이터 구축 비용 이외에도 label이 없기 때문에 **보다 일반적인 feature representation**과 **새로운 class가 들어와도 대응이 가능** 하다는 장점이 추가적으로 존재한다.

이후 classification 등 다양한 downstream task에 대해서 네트워크를 fine-tuning 시키는 방향으로 활용하곤 한다.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/contrastive_learning/fig1.png" alt="Fig 1" style="width:90%;">
  </div>
  <figcaption style="text-align:center">Fig 1. Feature를 학습한 이후의 활용</figcaption>
</figure>

## Contrastive Representation Learning
Representation Learning은 크게 2가지 접근법이 존재한다.
하나는 생성모델의 측면 나머지는 판별모델의 측면이다. 
**생성모델**로 데이터의 표현을 학습하는 경우, **비지도 학습이기 때문에 데이터 구축 비용이 낮다는 장점**이 있다. 또한 저차원 표현을 학습하는 데 있어 **목적함수가 보다 일반적**이라는 장점이 있다.

**판별모델**의 경우에는 **계산 비용이 적고, 학습이 용이**하다는 장점이 있다. 대부분 라벨링된 데이터에 의존하기 때문에 데이터 구축 비용이 크다는 단점이 있습니다. 판별 모델의 경우 데이터가 속한 클래스를 판별하는 목적을 지녔기 때문에, 보다 지엽적인 목적함수라고 할 수 있다. 실제로 판별모델을 학습하는 과정 중에 학습되는 representation은 texture에 보다 집중을 한다는 주장을 하는 [논문](https://arxiv.org/abs/1811.12231) 또한 발표되었다.

CRL도 representation learning을 수행하기 위한 하나의 방법이다. CRL은 앞서 말했듯이 입력 샘플 간의 비교를 통해 학습한다. 따라서, 목적은 심플하다. **학습된 표현 공간 상에서 비슷한 데이터는 가깝게, 다른 데이터는 멀게 존재하도록 표현 공간을 학습**하는 것이다.

여러 입력쌍에 대해서 유사도를 label로 판별 모델을 학습한다. 이때 유사함의 여부는 데이터 자체로부터 정의 될 수 있다. 즉 self-supervised learning이 가능하다.
<figure>
  <div style="text-align:center">
    <img src="/assets/img/contrastive_learning/fig2.png" alt="Fig 1" style="width:90%;">
  </div>
</figure> 

Contrastive 방법의 경우, 다른 task로 fine-tuning을 수행할 때에 모델 구조 수정 없이 이루어 질 수 있다는 점에서 훨씬 간편하다.

---





$$
\begin{equation}
\Delta_t := E(x_t) \\
w_{t+1} \leftarrow  \Delta_t + w_t
\end{equation}
$$

새로운 latent code $w_{t+1}$는 generator $G$를 통과하여 새로운 reconstruction의 예측값을 만든다. 

$$
\begin{equation}
\hat{y}_{t+1} :=  G(w_{t+1})
\end{equation}
$$

업데이트된 예측값 $\hat{y}_{t+1}$은 다시 입력 이미지 $x$와 concat되며 이 과정이 반복된다. 이 과정은 generator의 평균 style vector $w_0$와 generator로 합성한 대응되는 이미지 $\hat{y}_0$로 시작한다. 

단일 step으로 주어진 이미지를 invert하도록 인코더를 제한하면 학습에 엄격한 제약이 부과된다. 반면, ReStyle은 어떤 의미에서 이 제약을 완화하는 것으로 볼 수 있다. 위 식에서 인코더는 $w_0$를 이전 step의 output으로 guide하여 latent space에서 여러 step을 가장 잘 수행하는 방법을 학습한다. 이런 완화된 제약 조건에 의해 인코더는 자체 수정 방식으로 원하는 latent code로의 inversion의 범위를 좁힐 수 있다. 이는 최적화 방식과 비슷한 데, 최적화 방식과의 큰 차이점은 inversion을 효율적으로 수행하기 위해 인코더에서 step을 학습한다는 것이다. 

## Encoder Architecture

저자들은 ReStyle의 방식이 기존의 다양한 인코더 구조에 적용할 수 있다는 것을 보여주기 위해 state-of-the-art 인코더인 pSp와 e4e를 사용하였다. 이 두 인코더는 ResNet backbone에 Feature Pyramid Network를 사용하고 있으며 style feature를 세 개의 중간 레벨에서 추출한다. 이러한 계층적 인코더는 style input을 세 가지 레벨로 나눌 수 있는 얼굴 domain과 같은 잘 구조화된 domain에 적합하다. 이를 통해 이러한 디자인이 덜 구조화된 multimodal domain에 미치는 영향이 무시할 수 있지만 overhead가 증가한다는 것을 발견했다. 또한, ReStyle이 복잡한 인코더 구조의 필요성을 완화한다는 사실을 발견했다.

<center><img src='{{"/assets/img/restyle/restyle-encoder.PNG" | relative_url}}' width="60%"></center>

따라서 저자들은 pSp와 e4e의 더 단순한 변형 버전을 설계하였다. 인코더를 따라 세 개의 중간 레벨에서 style을 추출하는 대신 마지막 16x16 feature map에서만 style vector를 추출된다. $k$개의 style input이 있는 StyleGAN generator가 주어진다면, pSp에 사용된 map2style 블록을 $k$개 사용하여 feature map을 down-sampling하여 대응하는 512차원 style input을 얻었다. 

## Experiment
- Dataset: FFHQ (train) + CelebA-HQ (eval), Standford Cars, LSUN Horse & Church, AFHQ Wild
- Baseline: (인코더 기반) IDInvert encdoer, pSp, e4e / (최적화 기반) Karras / (Hybrid) 인코더 + 최적화
- Loss와 training detail은 pSp와 e4e의 기존 연구와 동일하게 사용
- $N=5$로 설정

<center><img src='{{"/assets/img/restyle/restyle-fig1.PNG" | relative_url}}' width="90%"></center>

<br>
각 데이터 셋에 대한 정량적 평가는 아래와 같다. 

<center><img src='{{"/assets/img/restyle/restyle-fig2.PNG" | relative_url}}' width="100%"></center>

<br>
다음은 각 step에서의 이미지 변화를 나타낸 그림이다.

<center><img src='{{"/assets/img/restyle/restyle-fig3.PNG" | relative_url}}' width="50%"></center>
<center><img src='{{"/assets/img/restyle/restyle-fig4.PNG" | relative_url}}' width="50%"></center>

<br>
위는 각 step에서 상대적으로 많이 변한 부분을 빨간색으로, 덜 변한 부분을 파란색으로 나타낸 것이다. 아래는 step 간의 상대적인 변화량을 나타낸 것이다. 

Editability 비교는 다음과 같다. 

<center><img src='{{"/assets/img/restyle/restyle-fig5.PNG" | relative_url}}' width="100%"></center>

<br>
다음은 입력 이미지에 대한 각 step의 output이다. 

<center><img src='{{"/assets/img/restyle/restyle-fig6.PNG" | relative_url}}' width="50%"></center>

## Encoder Bootstapping

<center><img src='{{"/assets/img/restyle/restyle-bootstrapping.PNG" | relative_url}}' width="60%"></center>

<br>
논문에서는 Encoder bootstrapping이라는 새로운 개념을 제시되었다. 먼저 FFHQ로 학습한 인코더로 step을 한 번 수행하여 latent code $w_1$과 대응되는 이미지 $\hat{y}_1$을 계산한다. 그런 다음 나머지 step은 Toonify 인코더로 latent code와 이미지를 계산하여 최종적으로 입력 이미지와 비슷한 Toonify 이미지를 만들어낸다. 

<p align="center">
  <img src='{{"/assets/img/restyle/restyle-fig7.PNG" | relative_url}}' width="45%">
  &nbsp;
  <img src='{{"/assets/img/restyle/restyle-fig8.PNG" | relative_url}}' width="45%">
</p>