import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder
    
    input : 
        n : 피쳐의 수입니다. (columns) [feature 몇 개 쓸래?]
        T : time step 입니다. (rows) [얼마나 넣을래?]
        m : encoder가 받고 반환할 hidden state size 입니다.
    """
    def __init__(self, num_features: int, time_steps: int, hidden_size: int):
        super(Encoder, self).__init__()
        
        self.n = num_features # n
        self.T = time_steps   # T
        self.m = hidden_size  # m

        # LSTM이 될 수도 있고 GRU가 될 수도 있습니다. 일단 논문대로 LSTM.
        self.lstm = nn.LSTM(
            input_size=self.n, 
            hidden_size=self.m, 
            dropout=0.2 # param - 나중에 arg로 바꿀듯
        )

        self.We = nn.Linear(
            in_features=self.T * 2 * self.m,
            out_features=self.T
        ) ## input : R^2m
        self.Ue = nn.Linear(
            in_features=self.T * self.T,
            out_features=self.T
        ) ## input : R^Tx1 [t시점의 k번째 column의 값들, x_tk]

        self.v_e = nn.Linear(self.T, 1) 


    def forward(self, x):
        """순서

        """
        batch_size = x.size()[0]

        # 가장 초기의 hs, cs
        hidden_state = torch.zeros(1, batch_size, self.m, device="cuda")
        cell_state = torch.zeros(1, batch_size, self.m, devcice="cuda")

        # 나중에 결과로 나올 애
        Encoding_output = torch.zeros(self.T, batch_size, self.m, device="cuda")

        # Attention Scores
        E = self.v_e(
            torch.tanh(
                self.We(#ht-1;St-1) + self.Ue(#xk) # TODO
            )
        ) 

        # Attention Weight
        attn_weight = torch.softmax(E, 1) 


        
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
    
    def forward(self, x):

        return x

class DARNN(nn.Module):
    def __init__(self):
        super(DARNN, self).__init__()

    def forward(self, x):

        return x