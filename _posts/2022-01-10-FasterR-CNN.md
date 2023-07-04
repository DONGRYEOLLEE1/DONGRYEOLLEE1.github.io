---
layout: post
title: Faster R-CNN
tags: [Faster R-CNN, Object Detection]
categories: CV
use_math: true
comments: true
---

본 포스팅은 기본적으로 [herbwood](https://herbwood.tistory.com/10)님의 글을 통해 작성되어졌으며 필자가 부가적으로 많은 다른 포스팅을 참고해 살을 덧붙였습니다.

<br><br>

## Abstract 

- Fast R-CNN은 Region Proposal 추출(CPU 기반) > 속도 저하라는 limitation 
- Detection을 end-to-end로 수행x 

- Faster R-CNN은 위의 두가지를 해결한 모델

<br><br>

## Faster R-CNN 모델
### Preview

![arctecture](/img/FasterR-CNN/image1.JPG)

위 그림과 같이 Faster R-CNN의 RPN은 Fast R-CNN이 어디에 주목해야하는지 알려준다.

RP를 추출하기 위해 사용되는 Selective Search 알고리즘은 위에 기술했듯이, CPU상에서 동작하고 이로 인해 네트워크에서 bottleneck현상이 발생하게된다. Faster R-CNN은 이러한 문제를 해결하고자 **RP 추출 작업을 수행하는 네트워크인 Region Proposal Network(RPN)**을 도입한다. RPN은 RP를 보다 정교하게 추출하기 위해 다양한 크기의 aspect ratio를 가지는 BB인 Anchor Box를 도입한다. RPN과 Fast R-CNN모델이 합쳐졌다고 볼 수 있다.

RPN에서 RP를 추출하고 이를 Fast R-CNN 네트워크에 전달하여 객체의 class와 위치를 예측한다. 이를 통해 모델의 전체 과정이 GPU상에서 동작하여 bottleneck 현상이 발생하지 않으며, end-to-end로 네트워크를 학습시키는 것이 가능해진다.

![RFPN](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FQerXG%2Fbtq2OP3guI2%2F7ThBkM2vLJfv0JdKbQU3T0%2Fimg.png)

1. 원본이미지를 pre-trained CNN 모델에 입력해 Feature Map를 얻음 (end-to-end로 학습)
2. Feature Map은 RPN에 전달되어 적절한 RP를 산출
3. RP와 1. 과정에서 얻은 Feature Map을 통해 RoI Pooling을 수행해 **고정된** 크기의 Feature Map을 얻음
4. Fast R-CNN에 3. 과정의 값을 입력해 Classification과 BB Regression을 수행

<br><br>

## 본론
### 1. Anchor Box

![densesampling](https://www.researchgate.net/profile/Baiying_Lei/publication/276253537/figure/fig9/AS:341208506355721@1458361855697/Illustration-of-dense-sampling-and-spatial-stacking-Features-are-sampled-densely-in-each.png)

![densesampling2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fs23Qe%2FbtqQJEyPGXs%2FHcxyf1zTjLC2cSYkiKxGQk%2Fimg.jpg)

Anchor box는 Sliding Window의 각 위치에서 Bounding Box의 후보로 사용되는 상자다. 이는 기존에 사용되던 Image/Feature Pyramids와 Multiple-scaled Sliding Window와 다음과 같은 차이를 보인다.

![ㅇㅇ](https://curt-park.github.io/images/faster_rcnn/Figure1.png)

동일한 크기의 sliding window를 이동시키며 window의 위치를 중심으로 사전에 정의된 다양한 비율/크기의 anchor box들을 적용하여 feature를 추출하는 것. 이는 image/feature pyramids처럼 image 크기를 조정할 필요가 없으며 filter 크기를 변경할 필요도 없으므로 계산효율이 높은 방식이라고 할 수 있다. 논문에선 3가지 크기(scale)와 3가지 비율(aspect ratio)의, 총 9개의 anchor box들을 사용했다.

SS를 통해 RP를 추출하지 않을 경우, 원본이미지를 일정 간격의 grid로 나눠 각 grid cell을 BB로 간주해 Feature Map에 encode하는 Dense Sampling 방식을 사용한다. 이같은 경우 sub-sampling ratio를 기준으로 grid를 나누게 된다. 가령 원본 이미지가 800x800이며, sub-sampling ratio가 1/100이라고 할 때, CNN모델에 입력시켜 얻은 Feature Map의 크기는 8x8가 된다. 여기서 Feature Map의 각 cell은 원본 이미지의 100x100 만큼의 영역에 대한 정보를 함축하고 있다고 할 수 있다. 원본 이미지에서는 8x8개 만큼의 BB가 생성된다고 볼 수 있다.

![anchorbox](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FZaxPg%2FbtqQIaSDb3s%2FwfOr4FA6CxKGCgTDMkmkRk%2Fimg.jpg)

하지만 이처럼 고정된크기의 BB를 사용할 경우, 다양한 크기의 객체를 포착하지 못할 수 있다는 문제가 있다. 논문에선 이 문제를 해결하고자 지정한 위치에 사전에 정의한 서로 다른 크기와 종횡비를 가지는 BB인 Anchor Box를 생성해 다양한 크기의 객체를 포착하는 방법을 제시한다. 논문에선 각각 3개씩 즉, 9개의 서로 다른 Anchor box를 사전에 정의한다.

$w \times h = s^2$

$w =  \frac{1}{2} \times h$

$\frac{1}{2} \times h^2 = s^2$

$h = \sqrt{2s^2}$

$w = \frac{\sqrt{2s^2}}{2}$


여기서 scale은 anchor box의 width($w$), height($h$)의 길이를 aspect ratio는 width, height의 길이의 비율을 의미한다. aspect ratio에 따른 width, height의 길이는 aspect ratio가 1:1일 때의 anchor box의 넓이를 유지한 채 구한다.


![anchorbox](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fber1jG%2FbtqQIWfpujO%2FLUdR8MxjKxwuWUmnjcEqQk%2Fimg.png)

anchor box는 원본 이미지의 각 grid cell의 중심을 기준으로 생성한다. 원본 이미지에서 sub-sampling ratio를 기준으로 anchor box를 생성하는 기준점인 anchor를 고정한다. 이 anchor를 기준으로 사전에 정의한 anchor box 9개를 생성한다. 위의 그림에서 원본 이미지의 크기는 600x800이며, sub-sampling ratio = 1/16이다. 이 때 anchor가 생성되는 수는 1900(600/16 x 800/16)이며, anchor box는 총 17100개가 생성된다. 이같은 방식을 사용할 경우, 기존에 고전된 크기의 BB를 사용할 때보다 9배 많은 BB를 생성하며, 보다 다양한 크기의 객체를 포착하는 것이 가능해진다.

이 Anchor box의 수는 이론적으론 물체마다 제한이 있지만, 실제론 높은 값으로 사용 가능하다.

#### Anchor Box를 사용한 이유??
- Translation-Invariant(이동불변성)   
  - RPN에서 Window 사이즈를 Sliding하는 방식은 이동불변성 보장
  - CNN에서 Sliding Window를 사용해 Convolution을 하였을때 얻는 효과와 동일
  - 모델 사이즈를 줄여준다. 
  - 이미지안에 어떤한 클래스를 갖던간에 특징을 뽑아내는 이동불변성이라는 큰 장점을 가진다

- Multi-scale Anchors as Regression Regerences
    - anchor의 파라미터 값을 다양하게 줌으로써 기존에 쓰였던 SPPNet이 사용했던 필터값을 상이하게 주는 효과와 동일한 결과를 얻을 수 있다.
    - 다양한 scale과 ratio를 활용한 anchor를 통해 효율적으로 계산할 수 있다.


<br>

### 2. RPN(Region Proposal Network)

![rpn11](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcMqq0O%2FbtqBFNcS3vP%2FTkehwBGKq51Tx7SwLHkt7k%2Fimg.png)


RPN은 Feature Map이 주어졌을때, 물체가 있을 법한 위치를 예측한다. k개의 anchor box를 이용하며, Sliding Window를 거쳐 각 위치에 대해 Regression과 Classification을 수행한다.

단순히 물체가 있는지 없는지에 대한 여부만 2개의 output으로 알려준다. 또한 물체가 존재하는 위치의 정보를 정확히 찾기 위해 $Reg$ layer를 거쳐 BB의 위치(중간점)를 더욱 잘 예측할 수 있도록 만든다.


RPN은 원본 이미지에 RP를 추출하는 네트워크이다. 원본 이미지에서 Anchor box를 생성하면 수많은 RP가 만들어진다. RPN은 **RP에 대해 class-score를 매기고, BB coefficient를 출력하는 기능을 한다.** RPN의 전체적인 동작과정은 다음과 같다

![RPNStructure](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fo7PTm%2FbtqAXir1rPy%2FVbzsfY9JMY9N3ixCe3zxb0%2Fimg.png)*RPN Structure*

좀더 직관적으로 설명된 그림을 보자

![RPN Structure2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb7xNNb%2FbtqAYHyrFDU%2FJDkko5dBYTMzZV96AcpakK%2Fimg.png)


1. 원본이미지를 pre-trained VGG 모델에 입력해 Feature Map을 얻는다. feature map의 크기는 H(height) x W(width) x C(channel)로 잡는다.

2. 위에서 얻은 Feature Map에 대해 3x3 conv 연산을 256 혹은 512 Channel만큼 적용한다. 위 그림의 intermediate layer에 해당하며, **이때 Feature Map의 크기 H, W가 유지될 수 있도록 padding을 추가한다.** 수행 결과, H x W x 256 혹은 H x W x 512 크기의 두번째 피쳐맵을 얻는다.

3. 두 번째 Feature Map을 입력받아 Classification과 Bounding Box Regression 예측값을 계산해주어야한다. 이때 FCL가 아니라 1x1 conv을 이용해 계산하는 Fully Convolution Network의 특징을 갖는다. 


4. 이때 출력하는 Feature Map의 channel수가 2x9가 되도록 설정한다. <font color = 'Red'>RPN에서는 후보영역이 어떤 class에 해당하는지까지 구체적인 분류를 하지 않고 객체가 포함되어 있는지 여부만을 분류</font>한다. 또한 anchor box를 각 grid cell마다 9개가 되도록 설정한다. 따라서 channel 수는 2(object 여부) x 9(anchor box 9개) 개가 된다.   
    이 과정의 연산은 가볍게 진행되어야하기때문에 오직 2개에 대한 score값을 매기는 것. (positive, negative)

5. BB Regressor를 얻기 위해 Feature Map에 대해 1x1 conv 연산을 적용한다. 이때 출력하는 Feature Map의 channel 수가 4(BB Regressor) x 9(Anchor box 9개) 개가 되도록 설정한다.

6. 앞서 얻은 값들로 RoI를 계산해야한다. 먼저 Classification을 통해서 얻은 물체일 확률 값들을 정렬한 다음, 높은 순으로 K개의 anchor만 추려낸다. 그 다음 K개의 anchor들에게 각각 Bounding Box Regression을 적용해준다. 그 다음 NMS를 적용해 RoI를 구해준다.

![result](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F7Plul%2FbtqQE9fOpFN%2FuK49JtrzjY7wvBZTfZhnWK%2Fimg.jpg)

RPN의 출력결과는 위 그림과 같다. 

좌측 표는 anchor box의 종류에 따라 객체 포함 여부를 나타낸 Feature Map이며, 우측 표는 Anchor box의 종류에 따라 BB Regressor를 나타낸 Feature Map이다. 이를 통해 8x8 grid cell마다 anchor box가 생성되어 총 576개의 RP가 추출되며, Feature Map을 통해 각각에 대한 객체 포함 여부와 BB Regressor를 파악할 수 있다.

이후 class-score에 따라 상위 N개의 RP만을 추출하고 NMS를 적용해 최적의 RP만을 Fast R-CNN에 전달하게 된다.

- 요약
   - 다양한 사이즈의 이미지를 입력값으로 object score와 object proposal을 출력
   - Fast R-CNN과 Convolution Layer를 공유
   - Feature Map의 마지막 Conv 층을 작은 네트워크가 Sliding하여 저차원으로 매핑
   - Regression과 Classification을 수행
   - <u>Fast R-CNN에서 Selective Search가 수행하는 역할을 Faster R-CNN에선 RPN이 수행</u>


<br>

### 3. Multi-task Loss

![수식](/img/FasterR-CNN/수식.JPG)

$i$ : mini-batch 내의 anchor의 index

$p_i$ : anchor $i$에 객체가 포함되어 있을 예측 확률

$p_i^*$ : anchor가 positive일 경우 1, negative일 경우 0을 나타내는 index 파라미터

$t_i$ : 예측 BB의 파라미터화된 좌표(coefficient)

$t_i^*$ : Ground Truth Box의 파라미터화된 좌표

$L_{cls}$ : Log Loss

$L_{reg}$ : Smooth L1 loss

$N_{cls}$ : mini-batch의 크기 (논문에선 256)

$N_{reg}$ : anchor 위치의 수

λ : balancing parameter (10으로 지정)

RPN과 Fast R-CNN을 학습시키기 위해 Multi-task Loss를 사용한다. 하지만 RPN에서는 객체의 존재 여부만을 분류하는 반면, Fast R-CNN에서는 배경을 포함한 class를 분류한다는 점에서 차이가 있다.

<br><br>

## Training Faster R-CNN
Faster R-CNN = RPN + Fast R-CNN이라고 설명하지만, 모델 내부에서 처리해야하는 다양한 작업들이 있어 상당히 복잡하다. 대표적으로 anchor를 생성하고 처리하는 작업과 적절한 RP를 추출하는 작업이있다. 

![Train](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdUGPmu%2FbtqQUY4zUul%2FtL5VqnJIEheFNFBoN0OtuK%2Fimg.png)

### 1. 사전학습된 VGG16모델로 특징 추출
사전학습된 VGG16 모델에 800x800x3 크기의 원본 이미지를 입력해 50x50x512크기의 Feature Map을 얻는다.
> sub-sampling ratio 는 1/16

- Input : 800x800x3 사이즈 이미지
- Process : 사전학습된 VGG16모델로 특징 추출
- Output : 50x50x512 크기의 Feature Map


### 2. Anchor generation layer로 Anchor 만들기

RP를 추출하기에 앞서 원본 이미지에 대해 anchor box를 생성하는 과정이 필요하다. 원본 이미지의 크기에 sub-sampling ratio를 곱한만큼의 grid cell이 생성되며, 이를 기준으로 각 grid cell마다 9개의 anchor box를 생성한다. 즉, 원본 이미지에 50x50(= 800x1/16 x 800x1/16)개의 grid cell이 생성되고, 각 grid cell마다 9개의 anchor box를 생성하므로 총 22500개의 achor box가 생성된다.

- Input : 800x800x3 사이즈 이미지
- Process : anchor를 만들어냄
- Output : 22500(50x50x9) 개의 anchor box

### 3. Class-score, BB Regressor by RPN

RPN은 VGG16으로부터 Feature Map을 입력받아 anchor에 대한 **class-score, BB Regressor를 반환**하는 역할을 한다.

- Input : 50x50x512 크기의 Feature Map
- Process : RPN으로부터의 RP
- Output : class-score(50x50x2x9 크기의 Feature Map)와 BB Regressor(50x50x4x9 크기의 Feature Map)


### 4. Proposal layer로부터의 Region Proposal

Proposal layer에서는 2번 과정에서 생성된 anchor box들과 RPN에서 반환환 class-score와 BB Regressor를 사용해 **RP를 추출하는 작업을 수행**한다. 먼저 NMS를 적용해 부적절한 객체를 제거한 후, class-score 상위 N개의 anchor box를 추출한다. 이후 Regression coefficients를 anchor box에 적용해 anchor box가 객체의 위치를 더 잘 detect하도록 조정한다.

- Input : 
    - 22500개의 anchor box
    - class-score(50x50x2x9 크기의 Feature Map)와 BB Regressor(50x50x4x9 크기의 Feature Map)
- Process : Proposal layer로부터의 RP
- Output : 상위 N개로부터 RP


### 5. anchor의 target layer로부터 RPN을 학습하는 동안 anchor를 선택

Anchor target layer의 목표는 **RPN이 학습하는데 사용할 수 있는 anchor를 선택**하는 것. 

먼저 2번 과정에서 생성한 anchor box 중에서 원본 이미지의 경계를 벗어나지 않는 anchor box를 선택한 후, positive/negative 데이터를 sampling 해준다. (positive = Object / negative = Background)

전체 anchor box 중에서 1) ground truth box와 가장 큰 IoU 값을 가지는 경우 2) ground truth box와의 IoU 값이 0.7이상인 경우에 해당하는 box를 positive sample로 선정한다. 반면 ground truth box와의 IoU값이 0.3이하인 경우에는 negative sample로 선정한다. 1)번의 기준만으로 아주 드물게 객체를 탐지하지 못하는 경우가 있어 후에 2)번의 기준이 추가되었다.

IoU 값이 0.3 ~ 0.7인 anchor box는 무시한다. 이러한 과정을 통해 RPN을 학습시키는데 사용할 데이터셋을 구성하게된다.

- Input : Anchor boxes, Ground Truth Boxes
- Process : RPN이 학습하는동안 Anchor를 선택
- Output : target regression coefficients와 함께 positive/negative 샘플링


### 6. Proposal Target layer로부터 Fast R-CNN을 학습하는동안 anchor를 선택

Proposal target layer의 목표는 Proposal layer에서 나온 RP 중에서 **Fast R-CNN 모델을 학습시키기 위한 유용한 sample을 선택**하는 것이다. 

여기서 선택된 RP는 1번 과정을 통해 출력된 Feature Map에 RoI Pooling을 수행하게 된다. 먼저 RP와 Ground Truth box와의 IoU를 계산해 0.5이상일 경우 Positive, 0.1 ~ 0.5 사이일 경우 negative sample로 라벨링된다.

- Input : 상위 N개의 RP, Ground Truth Boxes
- Process : Fast R-CNN이 학습하는동안 Anchor를 선택
- Output : target regression coefficients와 함께 positive/negative 샘플링


### 7. Max Pooling by RoI Pooling

원본 이미지를 VGG16 모델에 입력하여 얻은 Feature Map과 위 과정을 통해 얻은 sample을 사용해 RoI Pooling을 수행한다. 이를 통해 고정된 크기의 Feature Map이 출련된다. 

- Input : 
    - 50x50x512 크기의 Feature Map
    - target regression coefficients와 함께 positive/negative 샘플링
- Process : RoI Pooling
- Output : 7x7x512 크기의 Feature Map


### 8. Multi-task Loss를 통한 Fast R-CNN 학습

나머지 과정은 Fast R-CNN 동작 순서와 동일하다. 입력받은 Feature Map을 FCL에 입력해 4096크기의 Feature Vector를 얻는다. 이후 Feature Vector를 Classifier와 BB Regressor에 입력해 각각 (K+1), (K+1)x4 크기의 Feature Vector를 출력한다. 출력된 결과를 사용해 Multi-task Loss를 통해 Fast R-CNN 모델을 학습시킨다.

- Input : 7x7x512 크기의 Feature Map
- Process : 
    - FCL로부터 특징 추출
    - 분류기로 분류
    - BB Regressor로 BB Regression
    - Multi-task Loss로 Fast R-CNN 학습
- Output : Loss 값 도출(Reg loss + Smooth L1 loss)

<br><br>

## Alternating Training

전체 모델을 학습시키란 매우 어렵다. RPN이 제대로 RoI를 계산해내지 못하는데 뒷 단의 Classification 레이어가 제대로 학습될리가 없겠다. 그렇기에 Faster R-CNN논문에선 RPN과 Fast R-CNN을 번갈아가며 학습시키는 <font color = 'Red'>4-step Alternating Training</font>방법을 사용한다.

![A.T](https://www.researchgate.net/publication/330948715/figure/fig4/AS:723865001029632@1549594273165/The-method-of-alternating-training-regional-proposal-network-RPN-stage-and-fast-R-CNN.jpg)*[그림1]*

![A.T2](https://images.velog.io/images/skhim520/post/f01cc517-0c4f-4c88-9001-210623207aef/image.png)*[그림2]*

![A.T3](/img/FasterR-CNN/image3.JPG)


1. 먼저 Anchor Generation layer에서 생성된 Anchor box와 원본이미지의 Ground Truth box를 사용해 anchor target layer에서 RPN을 학습시킬 positive/negative 데이터셋을 구성한다. 이를 활용해 **RPN**을 학습시킨다. 이 과정에서 사전학습된 VGG16 역시 학습된다. (Train RPN)

2. Anchor Generation layer에서 생성한 anchor box와 학습된 RPN에 원본 이미지를 입력하여 얻은 Feature Map를 사용하여 Proposal layer에서 RP를 추출한다. 이를 Proposal target layer에 전달해 Fast R-CNN 모델을 학습시킬 positive/negative 데이터셋을 구성한다. 이를 활용해 **Fast R-CNN**을 학습시킨다. (Train Fast R-CNN using the proposals from RPN)

3. 앞서 학습시킨 RPN과 Fast R-CNN에서 RPN에 해당하는 부분만 학습(fine-tune)시킨다. 이 과정에서 두 네트워크끼리 공유하는 Convolutional layer, 즉 사전학습된 VGG16은 고정(freeze)한다. (Fix shared Convolutional layers and fin-tune unique layers to RPN)

4. 학습시킨 RPN(3번)을 활용해 추출한 RP를 활용해 Fast R-CNN을 학습(fine-tune)시킨다. 이때 RPN과 사전학습된 VGG16은 고정(freeze)한다. (Fine-tune unique layers to Fast R-CNN)

RPN과 Fast R-CNN을 번갈아가며 학습시키면서 공유된 Convolutional Layer를 사용한다. 하지만 실제 학습 절차가 상당히 복잡하여 이후 두 네트워크를 병합하여 학습시키는 Approximate Joint Training 방법으로 대체된다고 한다.

<br><br>

## Detection

![detection](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FFRm0S%2FbtqQUYQ3lUK%2FTMLKhSAkFbCFQiCfx6MOrK%2Fimg.png)

실제 Inference 때는 Anchor target layer와 Proposal target layer는 사용되지않는다. 두 layer 모두 네트워크를 학습시키기 위한 데이터셋을 구성하는데 사용되기 때문이다. Fast R-CNN은 Proposal layer에서 추출한 RP를 활용하여 Detection을 수행한다. 그리고 최종적으로 얻은 Predicted Box에 **NMS를 적용**하여 최적의 BB만을 결과로 출력한다.

<br><br>

## 의의

- Region Proposal을 Convolution network를 활용해 CPU에서 GPU의 영역으로 전환
- Fast R-CNN과 같은 Region-based Detector의 Feature Map을 Region-proposal Generating에도 사용
- RPN은 end-to-end로 학습이 가능하며, object 여부와 BB를 Regression하는 하나의 Full-connected Layer
- 빠른 속도와 높은 정확도의 Object Detection을 이루어냈다.

<br><br>

## 한계
- 속도가 한 이미지에 0.2초 걸리기에 1초에 5프레임(5fps) Detector가 Real-time Detector로 되기에는 역부족
- RoI Pooling에서 RoI가 stride로 항상 딱 떨어지진 않기에 픽셀 손실이 있을 수 있어 정교한 작업이 필요한 탐지 문제에는 더 좋은 방법이 필요해 보인다.
  
* [Source](https://nuggy875.tistory.com/33)

## Reference
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
* [[분석]Faster R-CNN](https://curt-park.github.io/2017-03-17/faster-rcnn/)
* [Faster R-CNN 논문(Faster R-CNN: Towards Real-Time ObjectDetection with Region Proposal Networks) 리뷰](https://herbwood.tistory.com/10)
* [[Object Detection] 3. Fast R-CNN & Faster R-CNN 논문 리뷰](https://nuggy875.tistory.com/33)
* [[ML] '더 빠른' Faster RCNN Object Detection 모델](https://techblog-history-younghunjo1.tistory.com/184?category=1031745)
* [[논문리뷰] Faster R-CNN 이해](https://cake.tistory.com/5)
* [갈아먹는 Object Detection [4] Faster R-CNN](https://yeomko.tistory.com/17?category=888201)
* [“Fast R-CNN and Faster R-CNN”](https://jhui.github.io/2017/03/15/Fast-R-CNN-and-Faster-R-CNN/)
* [From R-CNN to Faster R-CNN – The Evolution of Object Detection Technology](https://www.alibabacloud.com/blog/from-r-cnn-to-faster-r-cnn-the-evolution-of-object-detection-technology_593829)
* [Faster R-CNN for object detection](https://towardsdatascience.com/faster-r-cnn-for-object-detection-a-technical-summary-474c5b857b46)
* [Faster RCNN [1506.01497]](https://towardsdatascience.com/faster-rcnn-1506-01497-5c8991b0b6d3)
* [Region Proposal Network (RPN) — Backbone of Faster R-CNN](https://medium.com/egen/region-proposal-network-rpn-backbone-of-faster-r-cnn-4a744a38d7f9)