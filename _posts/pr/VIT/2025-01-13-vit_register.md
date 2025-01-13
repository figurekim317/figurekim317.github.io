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

## Introduction
