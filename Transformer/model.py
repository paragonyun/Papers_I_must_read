import torch
import torch.nn as nn
import copy

class Transformer(nn.Module) :
    '''
    기본적으로 Transformer를 정의하는 곳입니다.

    큰 그림의 구조만 여기서 정의해주고 encoder와 decoder는 
    따로 class로 지정해줍니다.
    '''

    def __init__(self, encoder, decoder) :
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def encode(self, x) : ## encoder의 forward 같은 곳입니다.
        out = self.encoder(x)
        return out


    def decode(self, z, c) : ## decoder의 forward 같은 곳입니다.
        out = self.decoder(z, c) ## decoder는 문장과 encoder에서 나온 context를 함께 받습니다.
        return out

    def forward(self, x, z) :

        c = self.encode(x)
        y = self.decode(z, c)

        return y


class Encoder(nn.Module) :
    '''
    내부적으로 Encoder Block이 n_layer 개 만큼 있는 ENCODER를 만들어줍니다.
    forward에선 만들어진 Layer를 돌면서 recurrent 합니다.
    '''

    def __init__(self, encoder_block, n_layer) :
        super(Encoder, self).__init__()
        self.layers = []

        for i in range(n_layer) :
            self.layers.append(copy.deepcopy(encoder_block)) 

    
    def forward(self, x) :
        out = x

        for layer in self.layers :
            out = layer(out)

        return out


class EncoderBlock(nn.Module) :
    '''
    Encoder 안에 들어갈 Encoder Block을 정의합니다
    self_attention을 거친 후에 position encoding과 feed forward를 거칩니다.
    '''

    def __init__(self, self_attention, position_ff) :
        super(EncoderBlock, self).__init__()

        self.self_attention = self.attention
        self.position_ff = position_ff

    
    def forward(self, x) :
        out = x
        out = self.self_attention(out)
        out = self.position_ff(out)

        return out





