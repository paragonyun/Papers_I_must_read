# GoogLeNet(2014)
[리뷰보기](https://blog.naver.com/paragonyun/222914679046)
## 핵심 요약
- 기존의 Dense Matrix보다 Neuro Science에 기반한 **Sparse Connection** 방법 제안
- Sparse Connection을 하는 경우, 더 효율적인 연산이 가능할 것으로 예상했지만 기존의 Computing 환경이 Dense에 맞춰졌기 때문에 살짝 변형된 방법을 사용했어야 했음
- _Correlation_ 이 높은 Nodes만 _Clustering_ 을 한 후에 이들을 Dense로 학습시키는 **Submatrix** 방법 사용

<br>

- 다양한 시각적 정보를 고려하기 위해 1x1, 3x3, 5x5를 병렬적으로 사용
- 이들은 모두 출력이 같게 나오기 때문에(Same Padding) 이들의 정보를 모두 Stack할 수 있다.
- 그러나 이렇게 하면 연산량이 굉장히 높게 나오고, 구글은 이것을 경계했기 때문에 3x3과 5x5 전에 1x1 을 넣어줌으로서 연산량을 줄였다.
- **_1x1 효과_** 
    1. Channel Reduction : RGB 3개의 채널의 정보를 Conv1d로 압축시켜준다.
    2. Efficient Using of Resource : Params가 비약적으로 줄어들면서 적은 컴퓨닝 자원으로도 좋은 성능을 낼 수 있게 해줌

<br>

- GoogLeNet은 굉장히 깊고 넓은 모델 구조를 가짐. 이에 Gradient가 끝까지 잘 전파되지 못하는 현상이 발견됨.
- 이를 방지하기 위해 중간에 2개의 Classifier를 삽입함. 이들을 논문에선 **Auxiliary Classifier**라고 함.
- 이들의 삽입으로 Gradient 소실을 방지하고 정규화 효과까지 노려볼 수 있었음
- 이들의 Loss는 0.3의 Weight를 주었고, Inference 때는 사용하지 않음.(Discarded)

<br>

<br>

## 핵심 단어들
- **Inception Module**
- **1x1 Conv Layer**
- **Sparsely Connected Architectures**
- **Auxiliary Classifier**

<br>

<br>

## 소감
개인적으로 1x1의 아이디어를 적극적으로 잘 확용한 논문이라고 생각한다. 사실 제일 인상 깊었던 부분은 Auxiliary Classifier를 활용하는 부분으로 Gradient Vanishing을 해결할 수 있는 획기적인 방법이라고 생각한다.   
모델을 보면 워낙 깊고 넓기 때문에 conv block과 inception block을 따로 구현해야했다. 그러나 논문 읽기의 즐거운 시작을 담당할 중요하고도 재밌는 논문이라고 생각한다. 