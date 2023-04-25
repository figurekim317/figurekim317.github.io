---
layout: post
title:  "[Paper review] Multi-Scale Context Aggregation by Dilated Convolutions (DilatedNet)"
date:   2017-05-26 15:05:55 +0300
image:  /assets/images/blog/post-5.jpg
author: uixgeek
tags:   
  - Paper review
  - Segmentation
  - Dilated convolutions
  - Computer Vision
  - AI
mathjax: true
use_math: true
---

# Multi-Scale Context Aggregation by Dilated Convolutions (DilatedNet) Review

- papers : https://arxiv.org/pdf/1511.07122.pdf

## 0. Abstract 

- Dense prediction problems are generally different from image classficiation.
- We propose a new Convolutional Network Module suitable for dense prediction problems.
- The proposed module, Dilated Convolution, integrates contextual information of various sizes without losing resolution.
- In particular, while increasing the receptive field exponentially, the resolution is not lost.
- Through the above method, we were able to achieve SOTA in the Semantic Segmentation field.

## 1. Introduction

- Semantic Segmentation is difficult because situations of various sizes must be inferred and classification in pixel units is required.
- To solve the above problem, DeconvNet repeated up-convolutions to infer situations of various sizes and restore the resolution.
- Another way is to take inputs of different sizes and combine them.
- However, these methods leave one question as to whether "Down sampling" and "Rescaled Image" are necessary.
- DilatedNet removes Down sampling and Rescaled Images to solve the above problem.
- And by combining several convolutions called Dilated, effective results are obtained without down sampling and rescaled input.

## 2. Dilated Convolutions

![Image for post](https://drive.google.com/uc?export=view&id=1f3IfdgpOVJWS6nUXpdMVYd3uzACqxuF7)

- The left side means normal convolution and the right side means dilated convolution.
- The biggest difference between convolution and dilated convolution is that the value l is attached to t in Kernel.
- By this value, both have the same 3x3 filter, but the Receptive Field differs between 3x3 and 5x5.

![Image for post](https://drive.google.com/uc?export=view&id=1geLZH_nPYp_OJ86gcI8ha--powr3D3A1)

![Definition of 2D convolution](https://drive.google.com/uc?export=view&id=1ZFdEsJz2mGTwCpNnHyOZ2GyEcRWULMBA)

- - Here, y means output, x means input, and h means kernel. Let's check what the above formula has for the example below.

![image-20210206174332859](https://drive.google.com/uc?export=view&id=1BmLeac42wHc0qbdesRYGunFxg2TZoCVp)

- output -13 comes out through the following process.

![image-20210206174345252](https://drive.google.com/uc?export=view&id=1-J0lNXXKzPxZAp5tNVoeVoq7ihdvSHml)

- Let's apply the same process once to the case where Dilated is 2.

![image-20210206174407172](https://drive.google.com/uc?export=view&id=1SUXQXCbReO3WoH994T1UPUw6st5Xom_R)

- The process of widening the receptive field through the above dilated process is as follows.

![image-20210206175311423](https://drive.google.com/uc?export=view&id=1j91Ij_0cpgY_hmvLmRjSECtro_BLB7Ao)

- (a) : F<sub>0</sub> (receptive filed, **green**) → "3x3 filter with 1-dilated convolution" (parameter 9, **red points**) → F<sub>1</sub>
- (b) : F<sub>1</sub> (receptive filed, **green**)→ "3x3 filter with 2-dilated convolution, padding 1" (parameter 9, **red points**) ≒ 7x7 filter → F<sub>2</sub>
- (c) : F<sub>2</sub> (receptive filed, **green**)→ "3x3 filter with 4-dilated convolution, padding 3" (parameter 9, **red points**) ≒ 15 x 15 filter → F<sub>3</sub>
- In case of convolution, the number of parameters increases linearly, but dilated convolution is efficient because it increases exponentially.
   - F<sub>i+1</sub> = 2<sup>i+2</sup> -1 * 2<sup>i+2</sup> -1 Receptive Field.



## 3. Multi-Scale Context Aggregation 

- Context module: This is to increase the performance of dense prediction structure by aggregating multi-scale contextual information.
- Input of context module: feature map with resolution of 64x64 through font-end (e.g. vgg16).
- Layer 1 ~ Layer 7 : 3x3 convolution with diffrent dilation is used.
- Layer 8: uses 1x1 convolution with 1-dilation.
- truncation: ReLU is used as the activation function after collation.

![image-20210204145337769](https://drive.google.com/uc?export=view&id=1C4kK5I__amtwTpil4IigX3r1ar4y1kLp)

- In the case of Receptive Field, since it is calculated centering on the original image, in the case of Layers 1 and 2, the size is different even if the Dilation is the same as 1. As shown in the figure below, in the case of Layer 2, it is applied to the data that has already turned the original image into a feature map.

![image-20210206184620570](https://drive.google.com/uc?export=view&id=1Gj676fyGBo4Bd890OvFVyVTfO1dMnct0)

- Initialization (Le, Quoc V., Jaitly, Navdeep, and Hinton, Geoffrey E. A simple way to initialize recurrent networks of rectified linear units. arXiv:1504.00941, 2015)

![image-20210206175709833](https://drive.google.com/uc?export=view&id=1pWsFGXrsBRd5uehjRilqaOztFwYrW_OY)

- **a** : the index of the input feature map
- **b** : the index of the output map

```python
                L.Convolution(
                    prev_layer,
                    param=[dict(lr_mult=1, decay_mult=1),
                           dict(lr_mult=2, decay_mult=0)],
                    convolution_param=dict(
                        num_output=num_classes * multiplier, kernel_size=3,
                        dilation=dilation, pad=dilation,
                        weight_filler=dict(type='identity',
                                           num_groups=num_classes,
                                           std=0.01 / multiplier),
                        bias_filler=dict(type='constant', value=0))))
```



## 4. Front END 

- a front-end prediction module: uses the VGG-16 network.
- Remove pooing and striding from the last two layers.
- The convolution operation of all layers except the last layer is applied 2-dilated.
- The convolution of the last layer is 4-dialed.
- You had to initialize all the previously learned weights while changing the convolution, but you can get a high-resolution output.

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1p9bfesQP_1pHtCtWjNm0qWho903kZOVH' />
    <figcaption><div style="text-align:center">Padding of 1 adds an extra layers on top of the input matrix. Left: Zero padding. Middle: Reflection padding. Right: Replication padding.
</div></figcaption>
</figure>



### Training 

- SGD
- minibatch size : 14
- learning rate : 10<sup>-3</sup>
- momentum : 0.9
- iteration : 60K

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1iaXiihE_GBsAFNURsNkGfLoWgBajXu_f' /><br>
    <figcaption><div style="text-align:center">Semantic segmentations produced by different adaptations of the VGG-16 classification network.
</div></figcaption>
</figure>

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1SiU_zFWNU2DbPqHFsqrYJVMmFSLTzo17' /><br>
    <figcaption><div style="text-align:center">Our front-end prediction module is simpler and more accurate than prior models. This table reports accuracy on the VOC-2012 test set.</div></figcaption>
</figure>



## 5. Experiments 

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1aIQt7PTuJwy2f6lI1m6gnsFctU9VSSbL' /><br>
    <figcaption><div style="text-align:center">Semantic segmentations produced by different models.</div></figcaption>
</figure>

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=19LgLEfY7To167AtbzhnTWyQT84A19ZFu' /><br>
    <figcaption><div style="text-align:center">Controlled evaluation of the effect of the context module on the accuracy of three different architectures for semantic segmentation.</div></figcaption>
</figure>

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1F6QoDinG0nb_xuTRQ8NU2PuuybJ3Vu5c' /><br>
    <figcaption><div style="text-align:center">Evaluation on the VOC-2012 test set. ‘DeepLab++’ stands for DeepLab-CRF-COCO-LargeFOV and ‘DeepLab-MSc++’ stands for DeepLab-MSc-CRF-LargeFOV-COCO-CrossJoint (Chen et al., 2015a).</div></figcaption>
</figure>



## 6. Conclusion 



Reference 

- https://daljoong2.tistory.com/181
- http://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5
- https://m.blog.naver.com/PostView.nhn?blogId=sogangori&logNo=220952339643&proxyReferer=https:%2F%2Fwww.google.com%2F

