import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

class PatchGenerator:
    """Patch를 만드는 곳입니다.
    torch.unfold를 이용하며 원본 이미지를 patch_size에 맞게 자릅니다.
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        num_channels = img.size(0) # (c, w, h)

        # w방향으로 먼저 자른다음, h방향으로 잘라줍니다.
        # unfold를 두번 거치게 되면 (3 x w/p x h/p x p x p)가 나옵니다.
        # 이들을 3 x wh/p^2 x p x p 형태로 바꿔주기 위해 reshape을 활용합니다.
        patches = img.unfold(1, self.patch_size, self.patch_size).\
                    unfold(2, self.patch_size, self.patch_size).\
                    reshape(num_channels, -1, self.patch_size, self.patch_size)
        
        # permute를 이용해 patch의 갯수인 wh/p^2을 맨 앞으로 빼줍니다.
        patches = patches.permute(1, 0, 2, 3)
        num_patch = patches.size(0)

        return patches.reshape(num_patch, -1)


class Flattened2Dpaches:
    def __init__(self, patch_size=16, img_size=256, batch_size=64):
        self.patch_size = patch_size
        self.img_size = img_size
        self.batch_size = batch_size

    def make_weight(self, labels, nclasses):
        """class수가 달라도 sampling되는 class를 동일하게 가져가기 위한 
        weight를 설정하는 곳입니다.
        ex) https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/
        쉽게 말해, 데이터를 sampling 할 때 균형있게 sampling하기 위한 weight를 만드는 곳입니다.
        """
        labels = np.array(labels)
        weight_arr = np.zeros_like(labels) # labels 만큼 빈 배열을 만들어줍니다.
        _, counts = np.unique(labels, return_counts=True) # 각 class 별로 몇 개의 value가 있는지 확인할 수 있습니다.
        for cls in range(nclasses):
            weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) # np.where로 조건에 맞는 index를 반환합니다. np.where(조건, True일 때, False 일때)

        return weight_arr

    def patchedata(self):
        """patch를 만들어 dataloader를 return 합니다."""
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomCrop(self.img_size, padding=2), # padding으로 crop을 해도 원본 사이즈가 유지되도록 합니다.
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Nomalize(mean, std),
            PatchGenerator(self.patch_size)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            PatchGenerator(self.patch_size)
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        # testset에서 valset을 걸러주기 위한 index 형성입니다.
        evens = list(range(0, len(testset), 2))
        odds = list(range(1, len(testset), 2))

        # 짝수만 valset으로 씁니다.
        valset = torch.utils.data.Subset(testset, evens)
        testset = torch.utils.data.Subset(testset, odds)

        weights = self.make_weight(trainset.targets, len(trainset.classes)) # Weight Sampling을 위한 Weight Array를 만듭니다.
        weights = torch.DoubleTensor(weights) # Tensor로 만들어줍니다.
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) # 가중치가 적용된 sampler 생성
        
        trainloader = DataLoader(trainset, batch_size=self.batch_size, sampler=sampler)
        valloader = DataLoader(valset, batch_size=self.batch_size, sampler=sampler)
        testloader = DataLoader(testset, batch_size=self.batch_size, sampler=sampler)

        return trainloader, valloader, testloader

def imshow(img):
    """시각화를 위한 함수입니다."""
    plt.figure(figsize=(100, 100))
    plt.imshow(img.permute(1,2,0).numpy())
    plt.savefig("./patch.png")
    plt.show()

if __name__ == "__main__":
    """이거만 단독으로 실행 시켰을 때 잘 동작하나 테스트 해보기 위함입니다."""
    print("👀 잘 되나 함 보겠습니다...")
    batch_size = 64
    patch_size = 8
    img_size = 32
    num_patches =int((img_size*img_size) / (patch_size*patch_size))

    patch_cls = Flattened2Dpaches(img_size=img_size, patch_size=patch_size, batch_size=batch_size)
    train_loader, _ , _ = patch_cls.patchedata()
    
    imgs, labels = next(iter(train_loader))
    print(imgs.size(), labels.size())

    ## 하나만 꺼내와서 확인, flatten됐던 걸 다시 사각형으로 만들어야 하기 때문에 다시 reshape으로 만들어줍니다.
    sample = imgs.reshape(batch_size, num_patches, -1, patch_size, patch_size)[0]

    imshow(torchvision.utils.make_grid(sample, nrow=int(img_size/patch_size)))