---
layout: post
title:  Autonomy Computer Vision
#date:   2017-05-26 14:05:55 +0300
image:  /assets/images/blog/post-2.jpg
#author: uixgeek
tags:   CV, Object detection, Segmentation, Nvidia, Optimization, Quantization
---

Object detection 성능 개선을 위한 네트워크 구조 설계 및 최적화
- Yolo v5를 기반으로 하는 모델의 성능 개선 및 경량화를 위해 pruning 진행
- Class imbalanced problem을 해결하기 위하여 data의 overlap을 수치적 측정한 값    (effective number)을 loss function에 반영하는 방식을 통해 모델의 성능 개선

Segmentation model 성능 개선을 위한 네트워크 구조 설계 및 최적화
- DDRnet, DeepLab V3+, ESPNet 등 다양한 모델을 사용하고 자율주행 상황에   맞춰 사용하기 위해 모델 경량화 및 구조 최적화
- Static quantization, dynamic quantization, Quantization aware training    모두 적용하고 모델의 성능 비교
- Onnx 변환 및 최적화 진행
- Nvidia Xavier 환경에서 inference를 위해 TensorRT 최적화 진행

레거시 코드 리팩토링
- Gstreamer를 기반으로 하는 파이프라인 최적화
- SW decoding을 HW decoding으로 바꾸어 CPU overhead 줄임
- 다양한 센서의 입력을 HW decoding 모듈로 보내고 gstreamer의 모듈화하고   이를 plugin loader를 사용하여 딥러닝 모델이 로드하는 과정으로 개선

Semantic segmentation data를 이용한 weakly supervised instance segmentation model 개발