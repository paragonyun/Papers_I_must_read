# A Dual Stage Attention-based RNN for Time Series Prediction
[리뷰보기]()
## 핵심 요약
- DARNN은 Seq2Seq 기반 모델로, Encoder와 Decoder 각각 Attention을 적용한 모델이다.
- 시계열 문제를 해결하고자 할 땐 항상 2가지 이슈가 따른다.
    > 1. 긴 Sequence 중에서 어느 순간이 지금 이 순간과 가장 연관 있는지
    > 2. 어떤 Driving Series(변수)가 지금 결정을 내리는 데에 있어 제일 중요한지  

- Encoder에 있는 Attention은 `Input Attention`으로 Input 변수들 간의 연관성과 이전 Encoder의 hidden state를 동시에 고려한 결과물을 산출하는 역할을 한다.

- Decoder에 있는 Attention은 `Temporal Attention`으로 매 순간마다 Encoder의 어떤 hidden state가 가장 연관 있는지를 알아내는 역할을 한다.

- 즉, 기존의 시계열 모델들은 해봐야 시간의 연관성만 파악했다면, DARNN은 변수들 간의 영향력 또한 고려한 결과를 낼 수 있다는 것이다.

- DARNN은 Noise에 매우 강건하다. 원 변수 81개에 가짜 랜덤 변수(Noise) 81개를 추가로 하여 총 162개로 모델을 돌렸을 때, 변수의 중요도를 나타내는 attention weight가 원변수 81개에 가장 높게 분포하고, noise 변수에 대해서는 낮은 점수를 주었다. (개인적으로 제일 신기했다.)



<br>

_학습방법_  
- Adam and SGD  
- Batch Size : 128  
- LR : start from 0.01 and is reduced 10% after 10000 epoch  
- Loss Function : RMSE, MAE, MAPE  

_Parameters_
- Time Window(T) : 10
- Hidden Size(m=p) : 64 or 128

<br>

<br>

## 핵심단어들
- Input Attention
- Output Attention
- Driving Series
- Seq2Seq

<br>

<br>


## 소감
차원 계산에서 사실 많이 헷갈렸다. 그리고 n과 T의 차원이 사실 직관적으로 와닿지는 않아서 그림을 그려가며 Data의 흐름을 정리하면서 읽었다. 자세한 내용은 블로그를 참고하길 바란다.
그래도 데이터의 흐름을 차원 하나하나 고려해가면서 구현했기 때문에 다른 논문보다 구현하는 맛이 더 있었던 것같다.