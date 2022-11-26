from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk'])
vocab.set_default_index(vocab['unk'])

'''
여기서 활용되는 전처리 함수 설명


'''

def data_process(raw_text_iter) :
    '''
    input으로 들어온 text를 Tensor로 바꿔줍니다.
    '''
    data = [torch.tenor(vocab(tokenizer(item)), dtype=torch.long) \
            for item in raw_text_iter]

    return torch.cat(tuple(filter(lambda t : t.numel() > 0, data)))

## 위의 과정에서 train_ter가 이미 소모되었기 때문에 다시 한번 더 재정의 해줍니다.
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data  = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchfy(data, bsz) :
    '''
    들어온 단어를 bsz 만큼 자릅니다.
    '''

    seq_len = data.size(0) // bsz # 배치사이즈 만큼 나눠서 몇개씩 넣어야하나 확인
    data = data[:seq_len*bsz]
    data = data.view(bsz, seq_len).t().contiguous()

    return data.to(device)

def datasets(batch_size=20, eval_batch_size=10) :
    train_data = batchfy(train_data, batch_size)
    val_data = batchfy(val_data, batch_size)
    test_data = batchfy(test_data, batch_size)
    
    return train_data, val_data, test_data

def get_batch(source, i) :
    '''
    입력과 Target을 짝지어주는 함수입니다.
    '''
    bptt = 35

    seq_len = min(bptt, len(source)-1 -i)
    data = source[i: i+seq_len]
    target = source[i+1 : i+1+seq_len].reshape(-1) 

    return data, target
