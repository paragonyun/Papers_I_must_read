from dataset import CatVsDogImageFoler
from dataloader import return_dataloaders
from trainer import Train
from model import MyGoogLeNet
import torch.nn as nn
import torch
from torchsummary import summary

dataset = CatVsDogImageFoler()
train_dataset, val_dataset , test_dataset = dataset.dataset()

train_loader, val_laoder, test_loader = return_dataloaders(train_dataset, val_dataset, test_dataset)

model = MyGoogLeNet()

# configs
NUM_EPOCH = 1
CRITERION = nn.CrossEntropyLoss()
LR = 0.001
OPTIMIZER = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
# TODO
# LR Scheduler 구현

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

