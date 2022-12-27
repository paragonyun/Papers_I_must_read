import torch
import torch.nn as nn

class LinearProjection(nn.Module):
    """들어온 이미지에 대해 Linear Projection을 수행하는 class 입니다.
    class token과 함께 positional encoding도 함께 수행합니다.
    
    nn.Linear는 선형곱의 용도로 사용 가능합니다.
    nn.Parameter는 학습 가능한 파라미터로 바꾸는 함수입니다.
    tensor.repeat(배수)는 각 차원의 요소를 배수만큼 반복합니다.
    
    Transformer의 input을 만드는 곳입니다.
    """

    def __init__(self, patch_vec_size, num_paches, latent_vec_dim, drop_rate):
        super(LinearProjection, self).__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim) # D 만큼의 Vector로 만들어줍니다.
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim)) # 1xD 크기의 랜덤 파라미터 생성
        self.pos_embedding = nn.Parameter(torch.randn(1, num_paches+1, latent_vec_dim)) # 1 x N+1 x D 형태의 positional embedding을 생성합니다.
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0) # (b, N, p^2c) 가 들어옴
        x = torch.cat([
            self.cls_token.repeat(batch_size, 1, 1), # class token을 모든 x에 적용하려면 b만큼이 더 필요하므로 repeat로 만들어줍니다.
            self.linear_proj(x)
        ], dim=1)
        x += self.pos_embedding
        out = self.dropout(x)
        return out 

class MultiheadedSelfAttention(nn.Module):
    """Multi Head Attention을 게산하는 class입니다.
    
    Dh = D/k로 정의한 후, D = Dh*k 이라는 사실을 이용하여 
    Linear 연산을 통해 한번에 여러개의 head 연산이 가능하도록 합니다.

    이렇게 얻어진 k개의 head의 q, k, v는 이후 view 함수를 통해 각 head의 것으로 쪼개어집니다.

    @ 는 torch.matmul()과 같은 역할입니다.    
    """
    def __init__(
        self, 
        latent_vec_dim,
        num_heads,
        drop_rate
    ):
        super(MultiheadedSelfAttention, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads) ## Dh = D/k 로 정했었음(논문에서)
        
        # 가장 중요한 부분입니다.
        # 원래는 output의 크기가 Dh로 나와야합니다.
        # 그러나 D는 그 자체로 k*Dh 를 의미하는데 이는 동시에 각 헤드를 k 번 반복한다는 의미와 같습니다.
        # 때문에 그 크기인 D를 그대로 output으로 놔둬도 multi-headed attention 연산을 수행할 수 있습니다.
        # 즉, 아래의 q, k, v는 모든 헤드의 q, k, v를 구한 것과 같습니다.
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)

        self.scale = torch.sqrt(self.head_dim*torch.ones(1).to(device)) # 그냥 tensor만 생성하면 cpu로만 가능합니다. 때문에 gpu로 한번 옮겨줘야 합니다.
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # head 마다의 q, k, v를 산출하는 구간입니다.
        # D = k*Dh 이므로 view 함수를 통해 k와 dh로 쪼개어줍니다.
        # permute 함수는 heads를 앞으로 가져오기 위해 사용합니다.(q, k, v 는 헤드마다 계산 되어야 하기 때문에)
        ## 
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) # [permute] (b, N, Dh, K) -> (b, Dh, N, K) 
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # k.T로 미리 만들어주기 위해 위치를 바꿉니다. 여기서 나온 k는 사실상 kT입니다.
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)

        attention = torch.softmax(q @ k / self.scale, dim=-1) # 유사도를 구하는 연산입니다. ## -> (b, Dh, N, N)

        x = self.dropout(attention) @ v # dropout을 거친 뒤 v와도 matmul을 해줍니다. ## -> (b, Dh, N, k)
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim) # 모든 결과를 cocnat 시키기 위해 reshape을 사용합니다. -> (b, N, D) ## D = Dh*k

        return x, attention # layter별 attention score를 확인하기 위해 attention도 return해줍니다.


class TFencoderLayer(nn.Module):
    """하나의 블럭을 정의하는 class 입니다.
    위에서 정의한 Multi Head Attention 연산과 
    Layer Norm, 
    MLP 연산이 함께 이루어집니다.
    """
    def __init__(
        self, 
        latent_vec_dim,
        num_heads,
        mlp_hidden_dim, 
        drop_rate,
    ):
        super(TFencoderLayer, self).__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(
                                            latent_vec_dim=latent_vec_dim,
                                            num_heads=num_heads,
                                            drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(
                                nn.Linear(latent_vec_dim, mlp_hidden_dim),
                                nn.GELU(),
                                nn.Dropout(drop_rate),
                                nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                nn.Dropout(drop_rate)
        )

    def forward(self, x):
        z = self.ln1(x)
        x, attention = self.msa(z)
        z = self.dropout(z)
        x = x + z # Residual Connection을 수행해줍니다.
        z = self.ln2(x)
        z = self.mlp(z)
        out = x + z # 마찬가지로 Residual Connection 수행

        return out, attention


class ViT(nn.Module):
    """최종적으로 만들어뒀던 class들을 합치는 class 입니다."""
    def __init__(
        self, 
        patch_vec_size,
        num_patches,
        latent_vec_dim,
        num_heads,
        mlp_hidden_dim,
        drop_rate,
        num_layers,
        num_classes,
    ):
        super(ViT, self).__init__()
        self.patch_embedding = LinearProjection(patch_vec_size=patch_vec_size,
                                                num_paches=num_patches,
                                                latent_vec_dim=latent_vec_dim,
                                                drop_rate=drop_rate
                                                )

        ## num_layer 만큼 TFencoder layer를 생성하고 이어붙입니다. 
        ## 이를 마지막에  nn.ModuleList로 구성해줍니다.
        self.transformer = nn.ModuleList([
                            TFencoderLayer(latent_vec_dim=latent_vec_dim, num_heads=num_heads,
                                            mlp_hidden_dim=mlp_hidden_dim, drop_rate=drop_rate)
                                        for _ in range(num_layers)
                                        ])
        
        ## 마지막에 classes 추정을 위한 mlp 입니다.
        self.mlp_head = nn.Sequential(
                                    nn.LayerNorm(latent_vec_dim),
                                    nn.Linear(latent_vec_dim, num_classes)
                                    )

    def forward(self, x):
        att_lst = []

        x = self.patch_embedding(x)

        for layer in self.transformer:
            x, att = layer(x)
            att_lst.append(att)
        
        out = self.mlp_head(x[:, 0]) # 마지막의 cls token만 사용합니다.

        return out, att_lst