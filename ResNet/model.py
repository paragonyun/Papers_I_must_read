import torch
import torch.nn as nn


class ResNet(nn.Module) :
    '''
    ResNet Block을 쌓는 곳입니다.
    Basic Block과 BottleNeck Block을 선택하여 쌓을 수 있도록 했습니다.
    '''
    def __init__(self, config, output_dim, zero_init_residual=False) :
        super(ResNet, self).__init__()

        block, n_blocks, channels = config

        self.in_channels = channels[0]

        ## assert를 통해 4개가 아니면 에러가 뜨게 합니다.
        assert len(n_blocks) == len(channels) == 4 

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## 여기부터 Residual 이 시작됩니다. 여러개의 Block을 쌓는데
        ## 버전에 따라 Basic Block을 쌓을지, Bottle Neck Block을 쌓을지 정해주어야 하기 때문에
        ## get_renet_layer를 따로 정의해준 다음 추가해줍니다.
        self.layer1 = self.get_resnet_layer(block=block, n_blocks=n_blocks[0], channels=channels[0])
        self.layer2 = self.get_resnet_layer(block=block, n_blocks=n_blocks[1], channels=channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block=block, n_blocks=n_blocks[2], channels=channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block=block, n_blocks=n_blocks[3], channels=channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim) ## 마지막으로 나오는 채널수는 get_resnet_layer에 의해 in_channels가 됩니다.


        ## 일단 구현은 해놨는데, 쉽게 말해 각 블록의 마지막 BN을 0으로 초기화 시키는 함수입니다.
        ## 일단 저는 구현 자체가 목적이라 Pass 했습니다
        # if zero_init_residual :
        #     for m in self.modules() : ## module 내에서 정의된 layer들을 하나씩 iter로 반환합니다.
        #         if isinstance(m, BottleNeck) : ## m이 BottleNeck의 상속관계인지 확인합니다. 맞으면 True
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock) :
        #             nn.init.constant_(m.bn2.weight, 0)


    def get_resnet_layer(self, block, n_blocks, channels, stride=1) :
        layers = [] ## Blcok들이 들어갈 list입니다.

        ## Down Sampling을 해야하는지 결정하기 위한 if문 입니다.
        ## Down Sampling의 목적은 output으로 나온 게 input에 들어갈 수 없으니 shape을 맞춰주기 위한 용도이므로
        ## channel이 맞지 않으면 downsampling을 하고 맞으면 굳이 할 필요 없으니 pass 합니다
        if self.in_channels != block.expansion*channels : 
            downsample = True

        else :
            downsample = False

        ## Block을 Layer에 append 하는 곳입니다. block의 파라미터를 참고로 아래 주석으로 달아두겠습니다.
        ## Basic Block -> def __init__(self, in_channels, out_channels, stride=1, downsample=False) :
        ## BottleNeckBlock -> def __init__(self, in_channels, out_channels, stride=1, downsample=False) :
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks) : ## n_blocks-1만큼 반복합니다.
            layers.append(block(block.expansion*channels, channels))

        ## 다음 block을 고려해서 in_channels 값을 재정의합니다.
        self.in_channels = block.expansion*channels

        return nn.Sequential(*layers) ## list로 넣을 땐 이렇게 사용합니다.


    def forward(self, x) :
        x = self.conv1(x) # 224x224
        x = self.bn1(x) 
        x = self.relu(x)
        x = self.maxpool(x) # 112x112
        x = self.layer1(x) # 56x56
        x = self.layer2(x) # 28x28
        x = self.layer3(x) # 14x14
        x = self.layer4(x) # 7x7
        x = self.avgpool(x) # 1x1

        h = x.view(x.shape[0], -1)
        output = self.fc(h)

        return output, h



class BasicBlock(nn.Module) :
    '''
    기본적으로 사용되는 Block입니다.
    워낙 많이 사용되기 때문에 class로 그냥 정의해줍니다.
    ResNet34에선 BottleNeck이 사용되지 않기 때문에 관련된 파라미터인 expansion이 1로 고정되어 있습니다.
    그러나 BottleNeck이 있는 다른 버전(50, 101, 152)과 함께 사용되어야 하므로 없으면 안 되기 때문에 존재합니다.
    '''

    expansion = 1 ## Bottle Neck Define

    def __init__ (self, in_channels, out_channels, stride=1, downsample=False) :
        super(BasicBlock, self).__init__()

        self.basic_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        if downsample :
            '''
            DownSampling이 필요할 때 사용하는 곳입니다. 
            foward 함수를 보시면 아시겠지만, Identity를 더해주어야 할 때 shape이 달라 더할 수 없는 문제를 해결하기 위한 구간이므로
            1x1 Conv를 통해 channel을 조절합니다. 
            '''
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                            stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else :
            downsample = None
        
        self.downsample = downsample

    def forward(self, x) :
        i = x ## identity를 위해 처음 입력값을 복사해둔다고 생각하면 편합니다.
        x = self.basic_layer(x)

        if self.downsample is not None : ## else 이면 None으로 구현했기 때문에 downsampling인 경우에 i를 downsampling된 i로 대체합니다.
            i = self.downsample(i)

        x += i
        output = self.relu(x)

        return output



class BottleNeck(nn.Module) :
    '''
    1x1 Conv로 차원의 수를 조절하여 파라미터를 압도적으로 줄인 BottleNeck입니다.
    1x1의 kernel을 지나게 하면서 1차적으로 차원수를 줄인 상태에서 3x3 Conv2d 로 Feature를 추출하고
    추출이 되면 다시 1x1를 지나 이번엔 차원을 다시 늘려 residual이 원활하게 다시 연결될 수 있게 합니다.
    GoogLeNet의 아이디어가 계속해서 활용되고 있는 모습입니다.
    '''
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False) :
        super(BottleNeck, self).__init__()

        self.bottle_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                            stride=1, bias=False), ## 1x1 Conv 입니다. 차원수를 줄여줍니다.
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                            stride=stride, padding=1, bias=False), ## BottleNeck에서 Feature를 추출하는 3x3 Conv 입니다.
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(in_channels=out_channels, out_channels=self.expansion*out_channels,
                            kernel_size=1, stride=1, bias=False), ## 1x1 Conv 입니다. 여기선 차원수를 expansion을 곱한 만큼 증가시켜줍니다.
                nn.BatchNorm2d(self.expansion*out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        if downsample :
            ## BasicBlock과 마찬가지로 출력과 입력을 일치시켜주는 역할을 합니다.
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=self.expansion*out_channels, kernel_size=1,
                            stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

        else :
            downsample = None

        self.downsample = downsample

    def forward(self, x) :
        i = x
        x = self.bottle_block(x)


        if self.downsample is not None :
            i = self.downsample(i)
        x += i
        output = self.relu(x)

        return output

