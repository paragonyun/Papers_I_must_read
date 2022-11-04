# ResNet(2015)
[]  
## 핵심 요약
- 기존의 Network들은 깊게 쌓을 수록 좋다는 것은 알 수 있었으나, Gradient Vanshing과 같은 문제가 발생
- 이를 해결 하기 위해 이미 많은 논문에서 고민하였으나, 깊게 쌓으면 쌓을 수록 욓려 성능이 더 떨어지는 결과만 가지고 옴
- Microsoft 연구팀에서는 이를 **Identity Mapping**을 통해 해결함
- 가장 핵심이 되는 기법으로, Conv의 input값을 연산 결과인 output값에 더해주는 방법임
- 간단하지만 동시에 매우 _연산효율적_ 으로 parameter-free 라는 특성 때문에 Gradient도 input layer까지 잘 전달 되고, 학습 속도 측면에서도 매우 우수한 것으로 드러남.
- Identity Mapping을 사용하지 않는 Plain 버전의 모델과 비교했을 때, 층이 더 깊어졌음에도 성능이 더 우수해지는 것을 발견.

<br>

- 더해주는 과정에서 shape이 맞지 않는 문제는 **Down Sampling**으로 해결
- 동시에 너무 깊은 버전의 model은 기존의 구조를 **Bottleneck Block**으로 대체함으로서 연산 효율성을 증대시킴
- 이는 **1x1 Conv**의 채널을 원하는 대로 조절할 수 있다는 특성을 이용한 것으로, 이를 통해 shape을 조절하기도 함
- _학습 방법_
    - Augmentation : 224 crop, horizontal flip, per-pixel mean subtracted, standard color augmentation
    - Batch Normalization 사용
    - 가중치 초기화 사용
    - Optimizer 
        - SGD / 256 batch / lr : 0.1 - 에러가 높을 때 10% 수준으로 감소 / weight_decay : 0.0001 / momentum : 0.9
    - epoch : 60x10^4번
    - dropout을 사용하지 않음 (Identity의 성능만 보고 싶었기 때문)

<br>

<br>


## 핵심 단어들
- **Short Cut**
- **Residual**
- **Bottleneck**
- **Downsampling**
<br>

<br>


## 소감
간단하고 쉽게 구현 가능하지만 굉장히 기발한 아이디어로 훌륭한 성능을 이끌어낸 게 정말 놀랍다.  
GoogLeNet에 비해 shape을 맞추는 게 좀 까다롭긴 했지만 (Downsampling에서 좀 헷갈려서 버벅거렸다)  
구현하고 나니 코드적으로나 아이디어적으로 느낀 게 많았던 논문이라고 생각한다.