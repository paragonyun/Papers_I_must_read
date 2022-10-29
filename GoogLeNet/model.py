from math import ceil
from turtle import forward, st
import torch
import torch.nn as nn

class MyGoogLeNet(nn.Module) :
    def __init__(self, dim=64, num_cls=2, train_mode = True) :
        super(MyGoogLeNet, self).__init__()

        self.train_mode = train_mode

        # 논문에서의 시작 dim은 64입니다.
        # 제 테스트 데이터셋에서는 class가 2개입니다.
        # 처음의 Feature를 추출하기 위한 later입니다.

        ## Convolution Layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(dim, dim*3, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # 본격적으로 Inception Module이 나옵니다.
        # Inception 의 Paramter는 바로 아래의 Inception class를 참고하기 바라니다.
        # 순서대로 in_channels, 1x1, 1x1+3x3, 1x1+5x5, maxpool3+1x1 입니다.
        
        ## inception block NO.1
        self.inception_3a = Inception(dim*3, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(dim*4, 128, 128, 192, 32, 96, 64)
        self.Maxpool_3 = nn.MaxPool2d(3, stride=2, padding=1)

        ## inception block NO.2
        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # 끝까지 Gradient를 전달하기 위한 AuxClassifier입니다.
        self.aux_1 = AuxClassifier(512, num_cls=num_cls)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.aux_2 = AuxClassifier(528, num_cls=num_cls)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.Maxpool_4 = nn.MaxPool2d(3, stride=2, padding=1)

        ## inception block NO.3
        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)

        #AvgPooling을 마지막에 해줍니다.
        self.avgpooling = nn.AvgPool2d(kernel_size=7, stride=1)

        # dropout
        self.dropout = nn.Dropout2d(0.4)

        self.fc = nn.Linear(1024, num_cls)

    def forward(self, x) :
        x = self.conv_layer(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.Maxpool_3(x)
        
        x = self.inception_4a(x)
        # training_mode 일 때만 loss 추출
        if self.train_mode :
            aux1 = self.aux_1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.train_mode :
            aux2 = self.aux_2(x)
        x = self.inception_4e(x)
        x = self.Maxpool_4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.avgpooling(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        if self.train_mode :
            return [x, aux1, aux2]

        else :
            return x
        
        

        




class Inception(nn.Module) :
    '''
    인셉션 모듈을 정의하는 class 입니다.
    '''
    def __init__(self, ins, conv1_1, 
                            conv1_3, conv3_out,
                            conv1_5, conv5_out,
                            maxp3_1,
                                    ) :
        super(Inception, self).__init__()

        # 첫번째 1x1 branch 입니다.
        self.branch1_1 = nn.Conv2d(ins, conv1_1, kernel_size=1) 

        # 두번째 1x1 -> 3x3 branch 입니다.
        self.branch1_3 = nn.Sequential(
            nn.Conv2d(ins, conv1_3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv1_3, conv3_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # 세번재 1x1 -> 5x5 branch 입니다.
        self.branch1_5 = nn.Sequential(
            nn.Conv2d(ins, conv1_5, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv1_5, conv5_out, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # 네번째 maxpool(3) -> 1x1 branch 입니다.
        self.branchmax = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ins, maxp3_1, kernel_size=1, stride=1)
        )

    def forward(self, x) :
        branch1 = self.branch1_1(x)
        branch2 = self.branch1_3(x)
        branch3 = self.branch1_5(x)
        branch4 = self.branchmax(x)

        # 출력의 결과를 쌓아주는 부분입니다.
        fin_output = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        return fin_output


class AuxClassifier(nn.Module) :
    '''
    중간중간에 삽입된 Auxiliary Classifier를 구현한 class입니다.
    이들의 loss를 합치는 과정은 Train에서 합치거나 안 합치는 것으로 구현합니다.
    '''

    def __init__(self, ins, num_cls) :
        super(AuxClassifier, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.convlayer = nn.Sequential(
            nn.Conv2d(ins, 128, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(4*4*128, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_cls)
        

    def forward(self, x) :
        x = self.avgpool(x)
        x = self.convlayer(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        output = self.fc2(x)

        return output

        


