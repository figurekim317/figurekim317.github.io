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

## Introduction
객체 검출은 주어진 이미지에서 물체의 카테고리(클래스)와 위치(BBOX)를 동시에 진행하는 것이다.
Input에 대한 불균형 문제는 해당 속성의 대한 분포가 성능에 영향을 미친다. 결국 OD의 성능에 영향을 주게 된다.
OD에서 가장 일반적인 불균형 문제는 전경과 배경의 불균형으로 positive와 negative의 수가 극도로 불균형하다.

불균형 문제를 논문에서는 8가지 불균형 문제를 식별한다. 크게는 4가지의 불균형 문제로 분류
1. Class imbalance
- 서로 다른 클래스에 속하는 데이터의 양에 불균형이 있을 때 발생
- 전경과 배경의 불균형
- 클래스(positive)간의 불균형

2. Scale imbalance
- Input의 크기 불균형이 존재할 때 발생
- 객체들의 크기 불균형
- Feature들의 크기 불균형

3. Spatial imbalace
- 중심 위치, IoU같은 Box regression 과정에서 일어나는 문제
- Regression loss 불균형
- IoU 분포 불균형 (한 객체에 대해 너무 넓게 Box들이 분포)
- 객체 위치 불균형

4. Object imbalance
- 최소화 해야할 loss function이 너무 많을 때 발생
