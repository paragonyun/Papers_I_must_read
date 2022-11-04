from cmath import phase
import os
from utils import ImageTransform, train_val_split
from torchvision.datasets import ImageFolder


TRAIN_ROOT = r'mydata\train'
TEST_ROOT = r'mydata\test'

RESIZE = (224, 224) ## GoogLeNet Recommend Resolution
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

## ImageFolder를 사용할 경우
class CatVsDogImageFoler :
    def __init__(self) :
        self.train_dataset = ImageFolder(root=TRAIN_ROOT, transform=ImageTransform(RESIZE, MEAN, STD).data_transform['train'])

        self.train_dataset, self.val_dataset = train_val_split(self.train_dataset)

        self.test_dataset = ImageFolder(root=TEST_ROOT, transform=ImageTransform(RESIZE, MEAN, STD).data_transform['val'])

    def dataset(self) :
        return self.train_dataset, self.val_dataset , self.test_dataset

    '''
    d = CatVsDogImageFoler()
    tr, v , te= d.dataset()
    '''

# print(len(tr)) 346
# print(len(v))  39
# print(te)      98