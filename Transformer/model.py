import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np

class Transformer(nn.Module) :
    '''
    기본적으로 Transformer를 정의하는 곳입니다.

    큰 그림의 구조만 여기서 정의해주고 encoder와 decoder는 
    따로 class로 지정해줍니다.
    '''

    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator) :
        super(Transformer, self).__init__()

        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator


    def encode(self, src, src_mask) : ## encoder의 forward 같은 곳입니다.
        return self.encoder(self.src_embed(src), src_mask)


    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask) : ## decoder의 forward 같은 곳입니다.
        ## decoder는 문장과 encoder에서 나온 context를 함께 받습니다.
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def make_pad_mask(self, query, key, pad_idx=1) :
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

    def make_subsequent_mask(self, query, key) :
        """
        Masking을 해주는 곳, 다만 Token의 정답값을 알지 못하게 이후의 값을 Masking해줌
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        """
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('unit8')
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)

        return mask

    def make_tgt_mask(self, tgt) :
        """
        Encoder와 마찬가지로 Pad Mask를 해주는 부분
        """
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask

        return pad_mask & seq_mask


    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)

        return out, decoder_out




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
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]

    
    def forward(self, src, src_mask) :
        ## Mask를 외부에서 생성할 것이므로 인자로 받습니다.
        out = src

        # out에 Residual을 적용하기 위해 lambda 사용
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residuals[1](out, self.position_ff)
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


class PositionWiseFeedForwardLayer(nn.Module) :

    def __init__(self, fc1, fc2) :
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1
        self.relu = nn.ReLU() 
        self.fc2 = fc2

    def forward(self, x) :
        out = x 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ResidualConnectionLayer(nn.Module) :

    def __init__(self) :
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x, sub_layer) :
        out = x 
        out = sub_layer(out)
        out = out + x
        return out



"""Decoer
Decoer의 Input : Encoder로부터 나온 Context과 Sentence
                이때 Sentence는 Teacher Forcing으로 Label데이터에 해당하며
                모델이 학습의 방향을 잘 잡을 수 있도록 도와줌
"""




class Decoder(nn.Module):

    """
    두번째 Self-Attention Layer는 Input을 2개 받는다.
    1. Encoder에 넘어온 거 : K, V로 활용
    2. 이전 Self-Attention에서 온 거 : Q로 활용
    => 2의 1에 대한 Attention을 계산하는 역할!! 

    ** 목적 **
    - Teacher Forcing으로 넘어온 Sentencㄷ와 최대한 유사한 Predict Sentence 생성!!
    """

    def __init__(self, decoder_block, n_layer) :
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])


    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask) :
        out = tgt
        for layer in self.layers :
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)

        return out


class DecoderBlock(nn.Module) :

    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(3)]

    
    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask) :
        out = tgt
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out


class TransformerEmbedding(nn.Module) :

    def __init__(self, token_embed, pos_embed) :
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)

    def forward(self, x) :
        out = self.embedding(x)

        return out

class TokenEmbedding(nn.Module) :

    def __init__(self, d_embed, vocab_size) :
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x) :
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out


class PositionalEncoding(nn.Module) :
    def __init__(self, d_embed, max_len=256, device=torch.device('cpu')) :
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x) :
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out


def build_model(src_vocab_size, tgt_vocab_size, device=torch.device("cpu"), max_len=256, d_embed=512, n_layer=6, d_model=512, h=8, d_ff=2048):
    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(
                                    d_embed = d_embed,
                                    vocab_size = src_vocab_size)
    tgt_token_embed = TokenEmbedding(
                                    d_embed = d_embed,
                                    vocab_size = tgt_vocab_size)
    pos_embed = PositionalEncoding(
                                d_embed = d_embed,
                                max_len = max_len,
                                device = device)

    src_embed = TransformerEmbedding(
                                    token_embed = src_token_embed,
                                    pos_embed = copy(pos_embed))
    tgt_embed = TransformerEmbedding(
                                    token_embed = tgt_token_embed,
                                    pos_embed = copy(pos_embed))

    attention = MultiHeadAttentionLayer(
                                        d_model = d_model,
                                        h = h,
                                        qkv_fc = nn.Linear(d_embed, d_model),
                                        out_fc = nn.Linear(d_model, d_embed))
    position_ff = PositionWiseFeedForwardLayer(
                                            fc1 = nn.Linear(d_embed, d_ff),
                                            fc2 = nn.Linear(d_ff, d_embed))

    encoder_block = EncoderBlock(
                                self_attention = copy(attention),
                                position_ff = copy(position_ff))
    decoder_block = DecoderBlock(
                                self_attention = copy(attention),
                                cross_attention = copy(attention),
                                position_ff = copy(position_ff))

    encoder = Encoder(
                    encoder_block = encoder_block,
                    n_layer = n_layer)
    decoder = Decoder(
                    decoder_block = decoder_block,
                    n_layer = n_layer)
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
                        src_embed = src_embed,
                        tgt_embed = tgt_embed,
                        encoder = encoder,
                        decoder = decoder,
                        generator = generator).to(device)
    model.device = device

    return model