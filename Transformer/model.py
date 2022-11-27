import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math


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


    def encode(self, src, src_mask) : ## encoder의 forward 같은 곳입니다.
        out = self.encoder(src, src_mask)
        return out


    def decode(self, z, c) : ## decoder의 forward 같은 곳입니다.
        out = self.decoder(z, c) ## decoder는 문장과 encoder에서 나온 context를 함께 받습니다.
        return out

    def making_pad_mask(self, query, key, pad_idx=1) :
        # q : (n_batch, query_seq_len)
        # k : (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        # ne : 같은 위치에 있는 값들을 비교하여 다르면 True, 같으면 False 반환
        # tensor.ne(value) : 해당 Tensor와 비교했을 때, value면 False, 다르면 True 반환
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2) # unsqueeze로 차원을 더 만들어 줌
        key_mask = key_mask.repeat(1, 1, 1, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False

        return mask

    def make_src_mask(self, src) :
        pad_mask = self.make_src_mask(src, src)
        return pad_mask

    def forward(self, src, tgt, src_mask) :

        encoder_out = self.encode(src, src_mask)
        y = self.decode(tgt, encoder_out)

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

    
    def forward(self, src, src_mask) :
        ## mask를 받습니다.
        out = src

        for layer in self.layers :
            out = layer(out, src_mask)

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

    
    def forward(self, src, src_mask) :
        ## Mask를 외부에서 생성할 것이므로 인자로 받습니다.
        out = src
        out = self.self_attention(query=out, key=out, valu=out, mask=src_mask)
        out = self.position_ff(out)

        return out


class MultiHeadAttentionLayer(nn.Module) :
    def __init__(self, d_model, h, qkv_fc, out_fc) :
        super(MultiHeadAttentionLayer, self).__init__()

        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc)
        self.v_fc = copy.deepcopy(qkv_fc)
        self.out_fc = out_fc

    def forward(self, *args, query, key, value, mask=None) :
        '''
        각 input의 차원
        q, k v : (n_batch, seq_len, d_embed) ## 이 3개를 transform에 넣어 Q, K, V를 얻습니다.
        mask : (n_batch, seq_len, seq_len)
        return : (n_batch, h, seq_len, d_k)
        '''
        n_batch = query.size(0)

        def transform(x, fc) :
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)

            return out

        ## transform 함수에서 각각을 지정해준 fc를 거쳐 Q, K, V를 생성해줍니다.
        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        out = self.calculate_attention(query, key, value, mask) ## Attention Score를 계산합니다.
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        ## contiguous = 메모리 저장 상태를 axis 기준으로 하기 위함
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out)
        return out



def calculate_attention(query, key, value, mask) :
    '''
    self attention을 계산하는 곳입니다. 
    여러 개의 Q, K, V를 받으므로 batch의 수가 존재하며
    각 단계 뒤에 shape을 적었습니다.
    '''
    # q, k, v : (n_batch, seq_len, d_k) ## d_k : 임베딩의 차원 수
    # mask : (n_batch, seq_len, seq_len) ## mask matrix의 shape!!
    d_k = key.shape[-1]

    attention_score = torch.matmul(query, key.transpose(-2,-1)) ## 마지막 2개의 위치를 바꿔줍니다.
    attention_score = attention_score / math.sqrt(d_k)

    if mask is not None : ## 마스크가 존재하면...
        attention_score = attention_score.masked_fill(mask==0, -1e9) ## 0인 지점을 1e-9로 채웁니다.

    attention_prob = F.softmax(attention_score, dim=-1) # n_batch, seq_len, d_k -> d_k 차원의 값의 합을 1로 만들어줍니다.

    out = torch.matmul(attention_prob, value) # n_batch, seq_len, d_k

    return out


