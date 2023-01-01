# Vision Transformer
[리뷰보기](https://blog.naver.com/paragonyun/222971938804)
## 핵심 요약
- ViT는 기존에 사용되던 CNN을 전혀 사용하지 않고 Transformer의 Encoder 구조만 사용한 모델이다.
- 여기서 문제는 Attention을 어떻게 ViT에 사용할까인데, 이미지를 Patch로 만들어 일종의 Sequence 형태로 전달한다.
- 이렇게 Input 값을 Transformer가 받을 수 있는 형태로 바꾼 뒤엔, `Linear Projection`의 과정을 거치는 것 외에는 Transformer의 구조를 그대로 따르게 된다.
- `Linear Projection`은 Input으로 들어온 Patch들을 하나의 Latent Space에 사영하는 방법으로 단순 행렬곱으로 구현된다. 사영되는 차원은 D차원으로 표기하고 구현했다.
- ViT는 Cls Token을 이용하여 판단하는데, 이 Cls Token만 가져와서(1xD) 이를 Dx#cls 형태의 MLP에 넣어 각 class 별 확률을 구한다. 
- Architecture는 위와 같고, 큰 데이터일 수록 학습 성능이 좋으나 조금이라도 적으면 성능이 그렇게 좋지 못하기 때문에 실제로 사용될 때는 Large Dataset에 학습된 모델을 가져와 Fine Tuning 시키는 방식을 이용한다.


<br>

_학습방법_  

- 일반 학습
    - Adam
    - Linear LR Decay
        - weight decay : 0.1
    - Batch Size : 4,096
    - Label Smoothing
    - Early Stopping
- Fine Tuning
    - SGD
    - Cosine LR Decay
    - Grad Clipping
    - Batch Size : 512
    - Resizing
<br>

<br>

## 핵심단어들
- Patch
- Linear Projection
- Class Token
- Positional Embedding
- Inductive Bias

<br>

<br>


## 소감
이미지 Patch 만드는 거에 대해 고민이 많았는데 덕분에 Main Stream은 어떻게 하는지에 대한 감을 잡을 수 있었다. 다시봐도 Transformer가 가지는 Self-Attention 구조가 가지는 강력한 힘이 놀랍다. 이를 내 도메인 영역에 활용할 수 있는 용기? 를 얻은 것 같다. 더 열심히 공부해야겠다.