
import torch
import torch.nn as nn
import time
from tqdm import tqdm

from utils import calculate_acc


class Train :
    def __init__(self, 
                model,
                num_epoch,
                optimizer,
                criterion,
                tr_loader,
                val_loader,
                te_loader) :

        self.model = model
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.criterion = criterion
        self.tr_loader = tr_loader
        self.val_loader = val_loader
        self.te_loader = te_loader
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def training(self) :
        print('🚀 Start Trainiing... 🚀')
        print(f'Using Resource is... : {self.device}')
        best_acc = 0.0

        for epoch in range(1, self.num_epoch + 1) :
            start = time.time()
            
            train_loss, train_acc = self.train()
            val_loss, val_acc = self.eval(phase='valid')
                        
            it_takes = time.time() - start
            
            print(f'EPOCH {epoch}')
            print(f'TRAIN LOSS : {train_loss:.3f}\tTRAIN ACC : {train_acc*100:.2f}%')
            print(f'VALIDATION LOSS : {val_loss:.3f}\tVALIDATION ACC : {val_acc*100:.2f}%')
            print(f'It takes... {it_takes/60:.2f}m')
            print('=='*40)
            

            if val_acc > best_acc :
                best_acc = val_acc 
                print(f'\n✅ BEST MODEL IS SAVED at {epoch} epoch')
                torch.save(self.model.state_dict(), './BEST_MODEL.pt')

        print('\n\n')
        print('✨ Start Evaluation... ✨')
        test_loss, test_acc = self.eval(phase='test')
        print(f'TEST LOSS : {test_loss:.3f}\tTEST ACC : {test_acc*100:.2f}%')


    def train(self) :
        epoch_loss, epoch_acc = 0, 0
        self.model.train()
        self.model.train_mode = True
        self.model.to(self.device)

        for imgs, labels in tqdm(self.tr_loader) :
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            y_preds, _ = self.model(imgs)

            loss = self.criterion(y_preds, labels)
            acc = calculate_acc(y_preds, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss/len(self.tr_loader), epoch_acc/len(self.tr_loader)


    def eval(self, phase=None) : 
        epoch_loss, epoch_acc = 0, 0
        self.model.train_mode = False

        
        with torch.no_grad() :
            loader = self.val_loader

            if phase == 'test' :
                loader = self.te_loader
                self.model.load_state_dict(torch.load('./BEST_MODEL.pt'))

            self.model.eval()
            self.model.to(self.device)

            for imgs, labels in loader :
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                y_preds = self.model(imgs)
                
                loss = self.criterion(y_preds, labels)
                acc = calculate_acc(y_preds, labels)

                epoch_loss += loss.item()
                epoch_acc += acc.item()


        return epoch_loss/len(loader), epoch_acc/len(loader)

            


