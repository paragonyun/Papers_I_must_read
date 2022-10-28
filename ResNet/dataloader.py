from torch.utils.data import DataLoader

def return_dataloaders(tr_dataset, val_dataset, te_dataset) :

    BATCH_SIZE = 64

    train_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    te_loader =  DataLoader(te_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f'# of Train Datas : {len(tr_dataset)}')
    print(f'# of Validation Datas : {len(val_dataset)}')
    print(f'# of Test Datas : {len(te_dataset)}')

    return train_loader, val_loader, te_loader