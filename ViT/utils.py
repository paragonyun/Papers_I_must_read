import torch
import torch.nn as nn

def accuracy(dataloader, model):
    """중간중간 loss와 정확도를 체크하기 위한 함수입니다."""
    cor = 0
    total = 0
    running_loss = 0
    n = len(dataloader)
    
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval() # dropout을 비활성화 시키기 위함힙니다.
        for img, label in dataloader:
            imgs, labels = img.to(device), label.to(device)
            preds, _ = model(imgs)
            loss = criterion(preds, labels)
            _, predicted = torch.max(preds, 1)

            total += labels.size(0)
            cor += (predicted == labels).sum().item() # torch.eq로 해도 됩니다.
            running_loss += loss.item()

        loss_result = running_loss / n

    acc = 100*cor / total
    model.train() # 매 에폭마다 체크를 하고 싶어서 만든 함수기 때문에 마지막에 꼭 train모드로 바꿔줘야합니다.
    return acc, loss_result