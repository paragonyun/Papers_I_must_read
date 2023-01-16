import torch.nn as nn
import torch

"""
nn.Linear는 일종의 "학습 가능한 행렬"을 만들어준다. nxm Matrix에 bias를 더하고 말고까지 지정가능하다. 얼마나 좋은가!
n x m Matrix를 만들어주는데
in_features는 n
out_feature는 m이라고 생각하면 된다.

아래의 예시는 

128 x 20 형태의 Matrix가 들어왔을 때 20 x 30의 학습가능한 Matrix로 곱하는 연산의 예다. 
즉, ouptut은 행렬연산으로 인해 128 x 30이 나와야할 것이다.

단 AxB 연산을 수행할 때 B 위치에 있는 matrix를 만들어준다고 생각해야 편하겠지?
"""
linear = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = linear(input)
print(output.size()) # torch.Size([128, 30])

"""여기부턴 Encoder TEST"""
# Eq 8 되는지 테스트하기
m = 128
n = 10
T = 14

hs_1 = torch.randn(64, n, m)
st_1 = torch.randn(64, n, m)
concated = torch.cat([hs_1, st_1], dim=2)
x = torch.randn(64, T, n)

We = nn.Linear(in_features=2*m, out_features=T)
Ue = nn.Linear(in_features=T, out_features=T)
Ve = nn.Linear(in_features=T, out_features=1)


print("Hidden State : ", hs_1.size())
print("Cell State : ", st_1.size())
print("[ht-1;st-1] : ", concated.size())
print("input shape : ", x.size())

print(We(concated).size())
print(Ue(x.permute(0, 2, 1)).size())

print(Ve(We(concated) + Ue(x.permute(0, 2, 1))).size()) # 오!
attn_weight = torch.softmax(Ve(We(concated) + Ue(x.permute(0, 2, 1))), dim=1)
print("Attention Weight Size : ", attn_weight.size())
print("t시점의 x input : ", x[:, 0, :].size())
print("Final output")
print((x[:, 0, :] * attn_weight.squeeze()).size())
test_tensor = torch.Tensor([[[1,2,3],
                            [4,5,6],
                            [7,8,9]],
                            [[10,20,30],
                            [40,50,60],
                            [70,80,90]]])

print(test_tensor.size())

print(test_tensor[0])

"""여기부턴 Decoder TEST"""
T = 14
m = 128
p = 128
batch = 64
y_dim = 7

print("Input Size")
prev_y = torch.randn(batch, T - 1, y_dim)
encoding_output = torch.randn(T, batch, m)  # (T, b, m)
hs_1 = torch.zeros(1, batch, p)
cs_1 = torch.zeros(1, batch, p)
print("prev_y : ", prev_y.size())
print("Encoding Output : ", encoding_output.size())
print("hs_1 : ", hs_1.size())
print("cs_1 : ", cs_1.size())
print()

hs_concat = torch.cat((hs_1, cs_1), dim=2).repeat(T, 1, 1).permute(1, 0, 2)  # b, T, 2*p
print("concat [hs;cs] : ", hs_concat.size())

Wd = nn.Linear(in_features=2 * p, out_features=m, bias=False)
Ud = nn.Linear(in_features=m, out_features=m, bias=False)
Vd = nn.Linear(in_features=m, out_features=1, bias=False)


print("Wd(concat) : ", Wd(hs_concat).size())  # (T, b, m)
print("Ud(EncodingOutput) : ", Ud(encoding_output.permute(1, 0, 2)).size())  # (b, T, m)
print(
    "위에 꺼 두개 합", (Wd(hs_concat) + Ud(encoding_output.permute(1, 0, 2))).size()
)  # b, T, m
l = Vd(Wd(hs_concat) + Ud(encoding_output.permute(1, 0, 2)))
print("Vd(위에꺼 두개 합)", l.size())  # b, T, 1 ## 즉, T개의 l이 나오게 된 거임
print()

b_t = torch.softmax(l, dim=1)
print("b_t : ", b_t.size())  # b, T, 1

cv = torch.bmm(encoding_output.permute(1, 2, 0), b_t).squeeze(2)  # b, m, 1 - > b, m
print("Context Vector : ", cv.size())  # b, m

y2c = torch.cat((prev_y[:, 0, :], cv), dim=1)
print("t 시점의 y_hat : ", prev_y[:, 0, :].size())  # b, y_dim
print("y랑 cv랑 합친 거 : ", y2c.size())  # batch_size, y_dim+m

Ww = nn.Linear(in_features=y_dim + m, out_features=1, bias=True)
print("Ww(y2c)", Ww(y2c).size())  # batch_size, 1

hs = torch.randn(1, batch, p).squeeze(0)  # 원래의 shape -> squeeze -> b, p
d2c = torch.cat((hs, cv), dim=1)  # b, p+m
print()
print("d2c : ", d2c.size())

Wy = nn.Linear(in_features=p + m, out_features=p, bias=True)
Vy = nn.Linear(in_features=p, out_features=y_dim)
print("Wy(d2c) : ", Wy(d2c).size())  # b, p
print("Y_hat : ", Vy(Wy(d2c)).size())  # b, y_dim

"""torch.split
한 세트당 n개의 데이터를 가지게 나눕니다.
"""
arr = torch.tensor([[-1, -2, -3], 
                    [-4, -5, -6]])

result = torch.split(tensor=arr,
                    split_size_or_sections=2, dim=1)

print(arr.size())
print(arr)
print("쪼갠 결과")
print(result)
