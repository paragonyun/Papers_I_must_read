import model
import patchdata
import utils

import torch
import torch.optim as optim
import torch.nn as nn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--img_size', default=32, type=int, help='image size')
    parser.add_argument('--patch_size', default=4, type=int, help='patch size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--save_acc', default=50, type=int, help='val acc')
    parser.add_argument('--epochs', default=501, type=int, help='training epoch')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--drop_rate', default=.1, type=float, help='drop rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--latent_vec_dim', default=128, type=int, help='latent dimension')
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads')
    parser.add_argument('--num_layers', default=12, type=int, help='number of layers in transformer')
    parser.add_argument('--mode', default='train', type=str, help='train or evaluation')
    parser.add_argument('--pretrained', default=0, type=int, help='pretrained model')
    args = parser.parse_args()
    print(args)

    latent_vec_dim = args.latent_vec_dim # D의 크기를 의미합니다.
    mlp_hidden_dim = int(latent_vec_dim/2) # 보통 D/2로 한다고 합니다.
    num_patches = int((args.img_size * args.img_size) / (args.patch_size * args.patch_size)) # N = (w*h)/p^2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    patch_maker = patchdata.Flattened2Dpaches(patch_size=args.patch_size, img_size=args.img_size, batch_size=args.batch_size)
    train_loader, val_loader, test_loader = patch_maker.patchedata()
    
    patches, _, _ = iter(train_loader).next()

    # Model
    vit = model.ViT(
        patch_vec_size=patches.size(2),
        num_patches=patches.size(1),
        latent_vec_dim=latent_vec_dim,
        num_heads=args.num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        drop_rate=args.drop_rate, 
        num_layers=args.num_layers,
        num_classes=args.num_classes
    ).to(device)

    if args.pretrained == 1: ## Inference 할 때 사용합니다.
        vit.load_state_dict(torch.load('./BEST_MODEL.pth'))

    if args.mode == "train":
        print("🚀Start Training...🚀")
        criterion = nn.CrossEngropyLoss()
        optimizer = optim.Adam(vit.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

        n = len(train_loader)
        best_acc = args.save_acc

        for epoch in range(1, args.epochs + 1):
            running_loss = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                preds, _ = vit(imgs)
                loss = criterion(preds, labels)
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

                scheduler.step()

            train_loss = running_loss / n
            val_acc, val_loss = utils.accuracy(dataloader=val_loader, model=vit)

            if epoch % 10 == 0:
                print("===="*30)
                print("📃Training Report📃")
                print(f"EPOCH [{epoch}/{args.epochs}]\tVAL ACC {val_acc:.2f}")
                print(f"TRAIN LOSS {train_loss:.3f}\tVAL LOSS {val_loss:.3f}")
                
            if val_acc > best_acc :
                best_acc = val_acc
                torch.save(vit.state_dict(), './BEST_MODEL.pth')
                print("✔Best Model is Saved✔")
            
    else : ## Inference Mode
        print("✨TEST 시작...✨")
        test_acc, test_loss = utils.accuracy(dataloader=test_loader, model=vit)
        print(f"TEST LOSS {test_loss:.3f}\tTEST ACC {test_acc:.3f}")

