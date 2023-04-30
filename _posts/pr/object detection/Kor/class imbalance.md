---
layout: post
image:  /assets/images/blog/post-5.jpg
mathjax: true
title: "[Paper review] Imbalance Problems in Object Detection: A Review"
last_modified_at: 2023-04-28
categories:
  - Paper review
tags:
  - object detection
  - class imbalance
use_math: true
classes: wide
---

> IEEE TPAMI 2020. [[Paper](https://arxiv.org/abs/1909.00169)] [[Page]] [[Github]]  
> Kemal Oksuz, Baris Can Cam, Sinan Kalkan, Emre Akbas 
> 11 Mar 2020  

<center><img src='{{"/assets/img/stylegan-t/stylegan-t-fig1.PNG" | relative_url}}' width="50%"></center>

## 1. Introduction
객체 검출은 주어진 이미지에서 물체의 카테고리(클래스)와 위치(BBOX)를 동시에 진행하는 것이다.
Input에 대한 불균형 문제는 해당 속성의 대한 분포가 성능에 영향을 미친다. 결국 OD의 성능에 영향을 주게 된다.
OD에서 가장 일반적인 불균형 문제는 전경과 배경의 불균형으로 positive와 negative의 수가 극도로 불균형하다.

불균형 문제를 논문에서는 8가지 불균형 문제를 식별한다. 크게는 4가지의 불균형 문제로 분류
1.1. Class imbalance
- 서로 다른 클래스에 속하는 데이터의 양에 불균형이 있을 때 발생
- 전경과 배경의 불균형
- 클래스(positive)간의 불균형

1.2. Scale imbalance
- Input의 크기 불균형이 존재할 때 발생
- 객체들의 크기 불균형
- Feature들의 크기 불균형

1.3. Spatial imbalace
- 중심 위치, IoU같은 Box regression 과정에서 일어나는 문제
- Regression loss 불균형
- IoU 분포 불균형 (한 객체에 대해 너무 넓게 Box들이 분포)
- 객체 위치 불균형

1.4. Object imbalance
- 최소화 해야할 loss function이 너무 많을 때 발생

## 2. Background, Definition and Notation
2.1. SOTA in Object detection
- Object detection 두가지 주요 접근 방식
  - 상향식 (top-down) : 최근 제시, 나중에 처리 파이프라인에서 keypoint, parts 같은 하위 개체를 그룹화하여 object detection
    - 객체의 중요한 key-points (모서리, 중심점 등) 먼저 예측
    - key-points를 이용하여 전체 객체 인스턴스를 형성 및 그룹화
  
  - 하향식 (bottom-up) : 비교적 인기, object를 감지할 때 탐지 파이프라인 초기에 평가를 하는 방식
    - one-stage 모델과 two-stage 모델
      - Two-stage 모델
        - R-CNN, Fastet R-CNN 계열의 모델
        - sliding window를 통한 proposal mechanism을 사용하여 영역 검출(RoI)
        - RoI의 카테고리를 분류한 후 NMS를 통해 post-processing
      - One-stage 모델
        - SSD, YOLO, RetinaNet
        - 카테고리 분류와 Box 검출을 동시에 진행

2.2. Frequently used terms and Notation
- Feature Extraction Network/Backbone : 객체 검출시 Input image를 받는 Network
- Classification Network/Classifier : backbone에서 추출한 feature에서 분류 결과 까지 포함되며, confidence score를 표시
- Detection Network/Detector : classifier 와 regressor 두개를 포함하여 detection 결과로 변환
- Region Proposal Network (RPN) : 2-stage 모델에서 사용되며 backbone을 통해 생성된 proposal을 가지고 confidence score와 box coordinates를 포함
- Bounding Box : [x1,y1,x2,y2]가 일반 적이며 detection된 box의 정보를 나타냄
- Anchor : 2-stage 모델에서는 RPN에 1-stage 모델에서는 감지 부분에 사전에 정의된 Box 셋
- Region of Interest (ROI)/Proposal : RPN 에서 사용하는 proposal 메커니즘으로 생성된 박스 셋
- Input Bounding Box : detection network나 RPN에서 훈련시 사용하는 Anchor나 ROI 샘플
- Ground Truth : class, label, box등 사용자의 전처리된 데이터 셋
- Detection : (box정보, 클래스별 confidence score) 형식의 Output 데이터
- Intersection Over Union (IOU) : ground truth의 box와 detection output box와 겹침 정도 GIOU, CIOU, DIOU 등 여러 함수가 있다.
- Under-represented Class : 훈련시 데이터셋 또는 미니배치에 샘플이 적은 클래스 (클래스 불균형)
- Over-represented Class : 훈련시 데이터셋 또는 미니배치에 샘플이 많은 클래스 (클래스 불균형)
- Backbone Features : 백본에 적용하는동안 포함되는 feature set
- Regression Objective Input : 몇몇 방식에서는 log 도메인에서 Box를 직접 예측하게 변환하여 예측하는데, 이 때 명확성을 위해 모든 방법에 대한 regression loss 입력을 log에 표기함

## 3. Class Imbalace
- 배경(Background):
배경은 이미지나 영상에서 주요 관심 대상이 아닌, 물체 또는 사물들이 배치되어 있는 공간입니다. 일반적으로 배경은 이미지의 맥락을 제공하며, 주요 정보나 흥미로운 부분은 아닙니다. 배경은 대부분 멀리 떨어져 있거나, 덜 중요한 요소들로 구성되어 있습니다.
예를 들어, 사진에서 하늘, 나무, 건물 등은 배경으로 여겨질 수 있습니다.

- 전경(Foreground):
전경은 이미지나 영상에서 주요 관심 대상이 되는 물체 또는 사물들입니다. 이들은 이미지의 중심적인 부분에 위치하며, 관찰자의 주목을 끌고 주요 정보를 전달하는 역할을 합니다. 전경은 대부분 가까이에 있거나, 중요한 요소들로 구성되어 있습니다.
예를 들어, 사진에서 사람이나 동물, 중요한 물체 등은 전경으로 여겨질 수 있습니다.

Definition
- 과량의 background class와 소량의 foreground class는 훈련중 문제가 발생, background에 대한 labeling작업은 하지 않기에 class당 example 수에 의존하지 않는다.

Solution
- Hard sampling methods
  - OD에서 불균형 해결을 위해 자주 쓰이는 방법
  - cross-entropy의 계수를 0, 1로 이진화 시키는 것
    - positive와 negative 선택을 쉽게 만들며, 선택되지 못한 example을 모두 무시한다.
- Soft sampling methods
- Sampling-free methods
- Generative methods