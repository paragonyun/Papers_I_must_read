from dataset import CatVsDogImageFoler
from dataloader import return_dataloaders
from trainer import Train
from model import ResNet, BasicBlock, BottleNeck
import torch.nn as nn
import torch
from torchsummary import summary
from collections import namedtuple


dataset = CatVsDogImageFoler()
train_dataset, val_dataset , test_dataset = dataset.dataset()

train_loader, val_laoder, test_loader = return_dataloaders(train_dataset, val_dataset, test_dataset)

# model configs
ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

# resnet18_config = ResNetConfig(block=BasicBlock,
#                                 n_blocks=[2,2,2,2],
#                                 channels=[64,128,256,512])
# resnet34_config = ResNetConfig(block=BasicBlock,
#                                 n_blocks=[3,4,6,3],
#                                 channels=[64,128,256,512])

resnet50_config = ResNetConfig(block=BottleNeck,
                                n_blocks=[3,4,6,3],
                                channels=[64,128,256,512])

# resnet101_config = ResNetConfig(block=BasicBlock,
#                                 n_blocks=[3,4,23,3],
#                                 channels=[64,128,256,512])
# resnet152_config = ResNetConfig(block=BasicBlock,
#                                 n_blocks=[3,8,36,3],
#                                 channels=[64,128,256,512])
num_cls = 2
model = ResNet(resnet50_config, num_cls)

# train configs
NUM_EPOCH = 10
CRITERION = nn.CrossEntropyLoss()
LR = 1e-7
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LR)


print('Model Architecture 🚩')
print(summary(model, input_size=(3,224,224), device='cpu'))

trainer = Train(model=model, 
                num_epoch=NUM_EPOCH,
                optimizer=OPTIMIZER,
                criterion=CRITERION,
                tr_loader=train_loader,
                val_loader=val_laoder,
                te_loader=test_loader)

trainer.training()

