# Transformer
트랜스포머에 대한 내 생각부터 바꿨어야 했다. Transformer는 꼭 Seq에만 한정되는 것이 아니라,   
어떻게 **Encoding을 하느냐의 문제**이기 때문에 그 사용처는 NMT에서만 한정되지 않는다.  
- 입력 seq와 출력 seq의 길이는 다를 수 있다.  
- 입력 seq와 출력 seq의 도메인이 다를 수 있다.  
- 그러나 모델은 하나다.
    - RNN계열에선 n개의 단어가 들어가면 n 번 재귀가 일어나는데, 이건 그러지 않음. 한번에 n개를 인코딩 시킬 수 있음 (물론 생성할 땐 한 단어씩 만듦)
  
- 동일하지만 학습이 다르게 되는 Encoder와 Decoder 6개씩으로 구성됨

<br>

## n개의 단어가 어떻게 인코더에서 한번에 처리가 되는가?
**self-attention**이 그것을 가능하게 했다.  
1. n개 각각의 단어를 특정 숫자의 Vector로 나타낸다.
2. Self-attention은 n개의 단어를 n개의 Vector로 만들어주는 역할을 함
3. 중요한 것은 이 Vector는 입력된 다른 모든 단어를 고려한 Vector라는 것.  
    - 즉, x1에 대한 Vector z1을 만들 때, x1만 고려하는 것이 아닌 x2, x3, x4도 함께 고려해서 z1을 만든다!
4. Self-Attention은 따라서 Dependency가 있다고 할 수 있다.
  
이후 나오는 Feed Forward 신경망은 그저 이러한 입력을 처리하는 용도.

어떻게 처리하는지 다시 한번 예시로 살펴보자.
```
The animal didn't cross the street because it was too tired.
```
뒤의 It은 앞의 Animal을 뜻한다고 함.   
Transformer는 It이라는 단어를 Encoding할 때 다른 단어와의 관계를 표현하며 Embedding한다!!  

### 그걸 어떻게 하는가?
하나의 단어를 Vector로 만들 때, 3개의 Vector를 만듦(3가지 NN)
- Query, Key, Value (q, k, v) 
- 각각의 Vector를 통해 입력단어의 Embedding Vector를 새로운 Vector로 바꿔줌
- 그 다음의 과정은 아래와 같음.  

<br>

1. Score Vector를 만든다.
    - Score Vector : i번째 단어의 Query Vector와 나머지 단어들의 Key Vector를 내적함
    - 이는 두 Vector가 얼마나 "유사한지, 관계가 있는지"를 표현하는 것.  

2. Score Vector를 Normalize 해줌
    - 8로 나눈 다음 SoftMax로 바꿔줌 (여기는 사실상 Attention)  
    - Score의 크기가 너무 커지는 것을 방지하기 위함.

3. 이렇게 Normalized된 Score 값을 Value Vector와 곱해줌
    - 이렇게 되면 각 단어의 Value Vector는 각각의 Score와 곱해져서 더해짐
    - 이렇게 더해진 Vector가 바로 최종 Vector가 됨.
    
4. 이렇게 하면 나머지 현재 단어와 나머지 단어들을 모두 고려한 Vector를 얻어낼 수 있다.

<br>

> ❗ 주의할 점
> Query Vector와 Key Vector의 차원은 내적을 해야하기 때문에 항상 같아야 한다
> 그러나 Value Vector는 Weighted Sum의 역할만 하면 되기 때문에 조금 달라도 된다. (구현할 땐 편의상 같게함)
> 물론 마지막 출력으로 나온 해당 단어의 Vector 차원은 Value Vecotor와 같게 된다.
  
  <br>

말로 복잡하면 좀 복잡한데 이걸 행렬로 나타내면 좀 간단해짐  
위의 과정을 이해하고나면 행렬연산을 이해하는 것은 어렵지 않을 것!  
(그래서 코드로 구현하면 한 두 줄이면 끝나버림)

<br>

### **근데 이게 왜 잘 될까?**
기존의 방법들은 Input이 고정되어 있으면 Output도 고정됨.  
- 뭐 어쨌든 내부적으로 Weight로 고정되어있기 때문  
그러나 Transformer는 내 옆에 있는 단어에 따라서 Encoding이 달라짐
**즉, 입력이 고정되어 있더라도 옆에 있는 단어에 따라서 표현되는 것이 달라짐!!**
> 더 많고, 유연한 모델!!! 다양한 표현이 가능하다!! (그만큼 연산도 엄청 많이 든다...)
> 메모리를 많이 먹는다는 단점이 있긴 한데 그걸 상쇄하는 장점이 바로 유연성.

<br>

### Multi-headed Attention (MHA)
멀티헤드는 하나의 입력에 대해 위의 과정을 여러 개의 Attention으로 처리하는 것.  
즉, 하나의 Embedding Vector 입력에 대해 하나의 Q, K, V Vector만 만드는 게 아니라 여러개의 Q, K, V를 만드는 것!
- 이렇게 되면 1개의 Input에 대해 n개의 Encoding Vector가 나오게 된다.(8개)  

근데 이러면 다시 입력을 들어가기엔 차원이 안 맞으니까 n개의 z Vector들을 Wo라는 가중치를 또 곱해줘서 다시 다음 Encoder에 들어갈 수 있게끔 차원을 맞춰줌(1개짜리와 동일하게)

<br>

## ❗❗ Positional Encoding
입력에 특정 값을 더해주는 형태(Bias와 같음).  
> 생각해보면 Self-Attention의 작동 방식은 순서와 무관함
> **Sequential한 정보를 반영해주기 위해 사용!!!**  
구현도 보면 그냥 더해줌..  최근엔 좀 달라졌다고 함  

<br>

## Encoder와 Decoder 사이엔 어떤 정보를 주고 받는가?
Encoder들은 Key Vector와 Value Vector들을 보낸다.  
Input에 대한 단어들의 Attention Map을 Decoder에서 만드려면 Key Vector와 Value Vector가 필요하기 때문!!  
  
물론 학습할 때, Decoder는 Masking을 하게 됨. 이전 단어들만 보고 뒤에 있는 단어들은 참고하지 못하게 하기 위함!!(정답을 미리 알면 안 되니까)  
