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
    <img src="/assets/img/contrastive_learning/fig2.png" alt="Fig 2" style="width:90%;">
  </div>
</figure> 

Contrastive 방법의 경우, 다른 task로 fine-tuning을 수행할 때에 모델 구조 수정 없이 이루어 질 수 있다는 점에서 훨씬 간편하다.

---

CRL architecture의 하나인 Instance Discrimination Task (IDT)에 대해 설명을 하면 [Unsupervised Feature Learning via Non-Parametric Instance Discrimination(Zhirong Wu et al., 2018)](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0801.pdf)에서 처음 제안되었다.

IDT의 경우, Fig 3과 같이 네트웨크가 구성되고, 하나의 sample에서 두 가지의 view가 생성됨을 알 수 있다.
이때, 같은 이미지에서 나온(같은 인덱스에 위치한) pair는 무조건 positive pair이고, 그를 제외한 다른 인덱스 내의 view와는 모두 negative이다. pair의 구성은 다음과 같이 이루어진다.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/contrastive_learning/fig3.png" alt="Fig 3" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Fig 3. Contrastive Learning의 pair 구성</figcaption>
</figure>

Instance discrimination을 위한 contrastive learning의 architecture는 다음과 같이 구성된다.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/contrastive_learning/fig4.png" alt="Fig 4" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Fig 4. Contrastive Learning의 architecture</figcaption>
</figure>

### 1. Data Augmentation을 통한 input pair 생성

<figure>
  <div style="text-align:center">
    <img src="/assets/img/contrastive_learning/fig5.png" alt="Fig 5" style="width:80%;">
  </div>
</figure>

같은 이미지에서 생성되었다면 positive pair이고, pair 내 두 이미지가 다른 이미지로부터 나왔다면 negative pair이다.
Positive pair를 구성할 때는 원본 이미지에서 image transformation을 적용한 augmented image를 구성하여 pair를 구성하게 된다. 이때, augmentation (transformation)은 random crop, resizing, blur, color distortion, perspective distortion 등을 포함한다.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/contrastive_learning/fig6.png" alt="Fig 6" style="width:80%;">
  </div>
  <figcaption style="text-align:center">Fig 6. 다양한 augmentation 적용</figcaption>
</figure>

### 2. Generating Representation (= Feature Extraction)

입력 이미지 쌍을 생성했다면, 해당 이미지 쌍으로 representation을 학습(즉, 특징 추출)해야 한다.  
Contrastive learning network 내에서 이와 같은 부분을 **feature encoder** $e$라고 부르며,  
$e$는 아래와 같이 **특징 벡터** $v$를 출력하는 함수로 표현할 수 있다.

$$
e(\cdot) \rightarrow v = e(x), \quad v \in \mathbb{R}^d
$$

Encoder의 구조는 특정되지 않으며, 어떤 backbone network든 사용할 수 있습니다. 참고로 InstDisc에서는 ResNet 18을 사용했다.

---

### 3. Projection Head

**projection head** $h(\cdot)$에서는 encoder에서 얻은 특징 벡터 $v$를 더 작은 차원으로 줄이는 작업을 수행한다.  
간혹 여러 representation을 결합하는 방식으로 projection을 수행하기도 하는데, 이 경우에는 contextualization head라고도 지칭한다. 그러나 InstDisc에서의 projection head는 2048차원의 특징 벡터 $v$를 128차원의 metric embedding $z$로 projection하여, 즉 차원 축소를 수행하는 용도로 사용된다.

이때, projection head $h$는 다음과 같이 metric embedding $z$를 출력하는 함수로 표현될 수 있습니다.

$$
h(\cdot) \rightarrow z = h(v), \quad z \in \mathbb{R}^{'}, \quad d' < d
$$

Projection head의 경우엔 간단한 MLP 구조를 갖는다. 이후 unit vector로 정규화해준다.

#### metric embedding
Contrastive loss는 기본적으로 각 pair의 유사도를 측정한다. 이러한 유사도가 거리가 될 수도 있고, pair가 공유하는 entropy로 계산이 될 수도 있다. 즉, 유사도는 metric으로 나타낼 수 있고 이에 loss에 input으로 들어가는 z를 metric embedding이라고 표현하는 것이다. project head 내에서 feature representation space에서 metric representation space로 projection했다고 볼 수 있다.

### 4. Loss 계산
CRL의 목적(objective)은 positive pair의 embedding은 가깝게, negative pair의 embedding은 멀게하는 것이라고 말했는데 loss는 이러한 objective를 직접적으로 수행한다. 이를 contrastive loss로 부른다. Contrastive loss와 같은 경우에는 infoNCE, NTXent등이 많이 사용되고 있다.

- $i$번째 입력쌍에 대한 Loss의 일반항
$$
L = -\log \frac{\exp(z_i^T z'_i / \tau)}{\sum_{j=0}^{K} \exp(z_i^T z'_j / \tau)}
$$

- $z_i^T z'_i$: 두 벡터 $z, z'$의 내적. 여기서 $z'$는 $z$의 변형(transformation; augmented $z$).
- $\tau$: 하이퍼파라미터로, 두 벡터 간의 내적이 전체 loss에 어느 정도 영향을 미치는지 조절.
- 분모의 합($\sum$): $z_i$에 대해 하나의 positive pair와 $K$개의 negative pair를 포함하여 계산.
