import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder

    input :
        n : 피쳐의 수입니다. (columns) [feature 몇 개 쓸래?]
        T : time step 입니다. (rows) [얼마나 넣을래?]
        m : encoder가 받고 반환할 hidden state size 입니다.

    Data Shape : (batch_size x T x n)
    """

    def __init__(self, num_features: int, time_steps: int, hidden_size: int):
        super(Encoder, self).__init__()

        self.n = num_features  # n
        self.T = time_steps  # T
        self.m = hidden_size  # m

        # LSTM이 될 수도 있고 GRU가 될 수도 있습니다. 일단 논문대로 LSTM.
        self.lstm = nn.LSTM(
            input_size=self.n, hidden_size=self.m, dropout=0.2  # param - 나중에 arg로 바꿀듯
        )

        self.We = nn.Linear(
            in_features=2 * self.m, out_features=self.T, bias=False
        )  ## input : R^2m
        self.Ue = nn.Linear(
            in_features=self.T, out_features=self.T, bias=False
        )  ## input : R^Tx1 [t시점의 k번째 column의 값들, x_tk]

        self.v_e = nn.Linear(self.T, 1, bias=False)  # 연산할 때 Transpose되므로 T, 1로 둡니다.

    def forward(self, x):
        """순서"""
        batch_size = x.size()[0]

        # 가장 초기의 hs, cs,
        # 1은 나중에 Feature의 수인 n을 위한 곳입니다.
        hidden_state = torch.zeros(1, batch_size, self.m, device="cuda")
        cell_state = torch.zeros(1, batch_size, self.m, devcice="cuda")

        # 나중에 결과로 나올 애 (T, Batch_size, hidden_size)
        Encoding_output = torch.zeros(self.T, batch_size, self.m, device="cuda")

        for t in range(self.T):
            # concat hidden_state and cell_state
            hs = torch.cat((hidden_state, cell_state), dim=2)  # (1, batch_size, 2*m)

            # feature 수(n)만큼 공간을 만들어줌
            hs = hs.repeat(self.n, 1, 1).permute(1, 0, 2)  # (batch_size, n, 2*m)

            # Eqn (8) Attention Scores
            E = self.v_e(
                torch.tanh(self.We(hs) + self.Ue(x.permute(0, 2, 1)))
            )  # (batch_size, n, 1)

            # Attention Weight를 구함
            attn_weight = torch.softmax(E, dim=1)  # n이 dim=1에 있으므로 dim=1에 대해 softmax

            _, (hidden_state, cell_state) = self.lstm(
                (x[:, t, :] * attn_weight.squeeze()).unsqueeze(0),  # attention weight 를 곱해줌
                (hidden_state, cell_state),
            )

            Encoding_output[t] = hidden_state[0]  # 첫번 째 T에 대한 Encoding값을 넣습니다.

        return Encoding_output


class Decoder(nn.Module):
    def __init__(
        self,
        time_steps: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        pred_steps: int,
    ):
        super(Decoder, self).__init__()
        self.T = time_steps
        self.m = encoder_hidden_size
        self.p = decoder_hidden_size
        self.pred_steps = pred_steps

        ## l_ti를 구하기 위한 재료들 입니다.
        self.Wd = nn.Linear(in_features=2 * self.p, out_features=self.m, bias=False)
        self.Ud = nn.Linear(in_features=self.m, out_features=self.m, bias=False)
        self.Vd = nn.Linear(in_features=self.m, out_features=1, bias=False)

        ## y_t-1을 구하기 위한 재료들 입니다. (Ww : W물결표)
        self.Ww = nn.Linear(
            in_features=self.pred_steps + self.m, out_features=1, bias=True
        )  # 원래는 m+1인데 pred_steps 여러개 해야하므로 self.pred_steps 했습니다.

        ## f2 function
        self.lstm = nn.LSTM(1, self.p, dropout=0.2)

        ## y_hat을 구하기 위한 재료들 입니다.
        self.Wy = nn.Linear(in_features=self.p + self.m, out_features=self.p, bias=True)
        self.Vy = nn.Linear(self.p, self.pred_steps, bias=True)

    def forward(self, prev_y, Encoding_output):
        batch_size = Encoding_output.size(0)

        ## Encoder 때와 마찬가지로 초기의 Hidden State, Cell State를 설정해줍니다.
        hidden_state = torch.zeros(1, batch_size, self.p, device="cuda")
        cell_state = torch.zeros(1, batch_size, self.p, device="cuda")

        # Temporal Attention을 구하기 위한 과정입니다.
        # 이전 시점을 모두 싹 돌면서 Attention을 구하는 거기 때문에, 지난 모든 시점에 대해 반복합니다.
        for t in range(self.T - 1):
            # concat hidden_state and cell_state
            hs = torch.cat((hidden_state, cell_state), dim=2)  # (1, batch_size, 2*p)

            # 현 시점만큼 공간을 만들어줌
            hs = hs.repeat(self.T, 1, 1).permute(1, 0, 2)  # (batch_size, T, 2*p)

            hi = Encoding_output.permute(1, 0, 2)  # (batch_size, T, m)

            # Eqn 12
            l = self.Vd(
                torch.tanh(self.Wd(hs) + self.Ud(hi))  # (batch_size, T, m)
            )  # (batch_size, T, 1)

            # Eqn 13
            b_t = torch.softmax(l, dim=1)  # 모든 T 시점에 대해 Softmax
            # (batch_size, T, 1)

            # Eqn 14 (Context Vector)
            ## batch x batch 연산을 좀 원활하게 할 목적으로 bmm을 이용합니다.
            ## [B, n, x] x [B, x, m] = [B, n, m]
            cv = torch.bmm(
                Encoding_output.permute(1, 2, 0),  # (T, batch_size, m) -> (batch_size, m, T)
                b_t,  # (batch_size, T, 1)
            ).squeeze(
                2
            )  # -> (batch_size, m, 1) -> (batch_size, m)

            # Eqn 15
            ## 지금까지 Y의 t 시점과 cv로 y_wave를 얻습니다.
            y2c = torch.cat(
                (prev_y[:, t, :], cv), dim=1
            )  # (batch_size, pred_steps+m)
            Yy = self.Ww(y2c)  # (batch_size, 1)

            # Eqn 16
            ## 위의 결과를 LSTM에 넣어 최종 예측을 얻을 때 필요한 hidden state를 구합니다.
            _, (hidden_state, cell_state) = self.lstm(
                Yy.unsqueeze(0), (hidden_state, cell_state)  # (1, batch_size, 1)
            )

            ############ 여기까지가 Temporal Attention ############

        # Eqn 22
        d2c = torch.cat((hidden_state.squeeze(0), cv), dim=1)  # (batch_size, p + m)
        y_hat = self.Vy(self.Wy(d2c))  # (batch_size, p)  # (batch_size, pred_steps)

        return y_hat


class DARNN(nn.Module):
    def __init__(self,
                num_features: int, 
                time_steps: int,
                encoder_hidden_size: int,
                decoder_hidden_size: int,
                pred_steps: int
                ):
        super(DARNN, self).__init__()
        self.pred_steps = pred_steps

        self.encoder = Encoder(num_features=num_features, time_steps=time_steps, hidden_size=encoder_hidden_size)
        self.decoder = Decoder(time_steps=time_steps, encoder_hidden_size=encoder_hidden_size, decoder_hidden_size=decoder_hidden_size, pred_steps=self.pred_steps)

    def forward(self, inputs):
        X, Y = torch.split(
            inputs, [inputs.shape[2] - self.pred_steps, self.pred_steps]
        , dim=2)

        encoderoutput = self.encoder(X)
        y_hats = self.decoder(Y, encoderoutput)

        return y_hats
