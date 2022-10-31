from re import I
from turtle import down
import torch
import torch.nn as nn


# TODO
class ResNet(nn.Module) :
    def __init__(self, config, output_dim, zero_init_residual=False) :
        super(ResNet, self).__init__()

        







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
                nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        if downsample :
            ## BasicBlock과 마찬가지로 출력과 입력을 일치시켜주는 역할을 합니다.
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=self.expansion*out_channels, kernel_size=1,
                            stride=1, bias=False),
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

