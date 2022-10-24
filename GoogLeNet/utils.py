from torchvision import transforms 
from torch.utils import data
class ImageTransform :
    """
    이미지에 Augmentation을 적용하기 위한 Class
    """
    def __init__(self, resize, mean, std) :
        ## dictionary로 정의하여 train과 val을 따로따로 사용할 수 있게 만듦
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'val' : transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        }

    def __call__(self, phase) : 
        ## __call__ 메서드는 클래스가 호출 될 때 어떻게 할지 정할 수 있는 메서드입니다.
        
        # phase가 train이면 train에 지정된 transform을 수행하고, val이면 val로 지정된 transform을 수행합니다.
        return self.data_transform[phase]


def calculate_acc(y_pred, y) :
    '''
    정확도를 계산하기 위한 함수입니다.
    '''
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train_val_split(dataset) :
    '''
    Train data와 Valid data를 나누기 위한 함수입니다.
    '''
    TRAIN_RATIO = 0.9
    tr_ = int(len(dataset) * TRAIN_RATIO)
    val_ = len(dataset) - tr_

    train_dataset, val_dataset = data.random_split(dataset, [tr_, val_])

    return train_dataset, val_dataset


