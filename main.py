import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.nn.utils.prune as prune


device = 'cuda' if torch.cuda.is_available() else 'cpu'

from net import params_to_prune
from net import unfreeze_pruned_weights, unfreeze_pruning

class Trainer:
    def __init__(self,
                 optimizer,
                 model,
                 loss_func=None,
                 pretrain=False,
                 pretrained_pth=None):
        super().__init__()
        self.optimizer = optimizer
        self.loss_func = loss_func
        if pretrain:
            if not(pretrained_pth):
                raise Exception("Pretrained Path is None")
            else:
                model.load_state_dict(torch.load(pretrained_pth))
        self.model = model.train()

    def freeze_10(self):
        layers = [*self.model.model]

        for param in layers[0].parameters():
            param.requires_grad = False
        for param in [*layers[4]][0].conv1.parameters():
            param.requires_grad = False
        for param in [*layers[4]][0].conv2.parameters():
            param.requires_grad = False

    def freeze_20(self):
        layers = [*self.model.model]

        for param in layers[0].parameters():
            param.requires_grad = False
        for param in [*layers[4]][0].conv1.parameters():
            param.requires_grad = False
        for param in [*layers[4]][0].conv2.parameters():
            param.requires_grad = False
        for param in [*layers[4]][1].conv1.parameters():
            param.requires_grad = False
        for param in [*layers[4]][1].conv2.parameters():
            param.requires_grad = False

    def freeze_30(self):
        layers = [*self.model.model]

        for param in layers[0].parameters():
            param.requires_grad = False
        for param in [*layers[4]][0].conv1.parameters():
            param.requires_grad = False
        for param in [*layers[4]][0].conv2.parameters():
            param.requires_grad = False
        for param in [*layers[4]][1].conv1.parameters():
            param.requires_grad = False
        for param in [*layers[4]][1].conv2.parameters():
            param.requires_grad = False
        for param in [*layers[5]][0].conv1.parameters():
            param.requires_grad = False
        for param in [*layers[5]][1].conv2.parameters():
            param.requires_grad = False

    def pruning(self, amnt=0.1):
        pruned_params = params_to_prune(self.model)
        prune.global_unstructured(
            pruned_params,
            pruning_method=prune.L1Unstructured,
            amount=amnt
        )

    def unfreeze_pruned_weights(self):
        pruned_params = params_to_prune(self.model)
        for param in pruned_params:
            prune.remove(param[0], 'weight')

    def check_sparsity(self):
        pruned_params = params_to_prune(self.model)
        zero_weight_cnt = 0
        nelement_cnt = 0
        for param in pruned_params:
            zero_weight_cnt += torch.sum(param[0].weight == 0)
            nelement_cnt += param[0].weight.nelement()

        print(f'Sparsity of the network is :{100*zero_weight_cnt/nelement_cnt} %')

    def train(self, x, y):
        self.optimizer.zero_grad()
        self.output = self.model(x)
        self.gt = y
        loss = self.loss_func(self.output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def check_accuracy(self):
        total = self.output.size(0)
        self.predicted = torch.argmax(self.output, dim=1)
        correct = (self.predicted == self.gt).sum().item()
        return correct/total

class Tester:

    def __init__(self, model, loss_func=None):
        self.model = model.eval()
        self.loss_func = loss_func
        self.least_loss = float('inf')
        self.current_loss = None


    def validate(self, x, y):
        if self.loss_func == None:
            raise ValueError("Loss function can't be None for validation")
        self.output = self.model(x)
        self.gt = y
        loss = self.loss_func(self.output, y)
        self.current_loss = loss.item()
        return loss.item()

    def check_accuracy(self):
        total = self.output.size(0)
        self.predicted = torch.argmax(self.output, dim=1)
        correct = (self.predicted == self.gt).sum().item()
        return correct/total

    def save_model(self, filename):
        if self.least_loss > self.current_loss:
            self.least_loss = self.current_loss
            torch.save(self.model.state_dict(), filename)

    def test(self, x):
        return self.model(x)

if __name__ == '__main__':

    from data import dataloaders
    from net import resnet, resnet_for_cifar, params_to_prune
    from main import Trainer, Tester

    import os
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cifar_pth = '/.cifar'
    if os.path.exists(cifar_pth):
        os.mkdir(cifar_pth)

    resnet = resnet_for_cifar()
    resnet = resnet.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=1e-3)

    trainer = Trainer(optimizer=optimizer, model=resnet, loss_func=loss_func, pretrain=True, pretrained_pth='model.pth')
    trainer.pruning(amnt=0.1)
    trainer.unfreeze_pruned_weights()
    # trainer.freeze_10()
    # trainer.freeze_20()
    # trainer.freeze_30()
    tester = Tester(model=resnet, loss_func=loss_func)

    train_loss_per_epoch = []
    val_loss_per_epoch = []

    train_acc_per_epoch = []
    val_acc_per_epoch = []

    loader = dataloaders()
    train_dl = loader.return_traindl()
    val_dl = loader.return_testdl()

    best_training_acc = 0 # for a batch
    best_validation_acc = 0 # for a batch

    epochs = 600
    for epoch in tqdm(range(epochs)):

        train_loss_per_loader = []
        val_loss_per_loader = []

        train_acc_per_loader = []
        val_acc_per_loader = []

        for train_data, val_data in zip(train_dl, val_dl):
            train_img, train_label = train_data
            train_img, train_label = train_img.to(device), train_label.to(device)
            train_loss = trainer.train(train_img, train_label)
            train_acc = trainer.check_accuracy()

            val_img, val_label = val_data
            val_img, val_label = val_img.to(device), val_label.to(device)
            val_loss = tester.validate(val_img, val_label)
            val_acc = tester.check_accuracy()

            train_loss_per_loader.append(train_loss)
            val_loss_per_loader.append(val_loss)

            train_acc_per_loader.append(train_acc)
            val_acc_per_loader.append(val_acc)

            tester.save_model()

        train_loss_per_epoch.append(np.mean(train_loss_per_loader))
        val_loss_per_epoch.append(np.mean(val_loss_per_loader))

        avg_train_acc = np.mean(train_acc_per_loader)
        avg_val_acc = np.mean(val_acc_per_loader)

        if avg_train_acc > best_training_acc:
            best_training_acc = avg_train_acc

        if avg_val_acc > best_validation_acc:
            best_validation_acc = avg_val_acc

        train_acc_per_epoch.append(avg_train_acc)
        val_acc_per_epoch.append(avg_val_acc)

        with open('train_loss.npy', 'wb') as f:
            np.save(f, np.array(train_loss_per_epoch))
        with open('val_loss.npy', 'wb') as f:
            np.save(f, np.array(val_loss_per_epoch))

        with open('train_acc.npy', 'wb') as f:
            np.save(f, np.array(train_acc_per_epoch))
        with open('val_acc.npy', 'wb') as f:
            np.save(f, np.array(val_acc_per_epoch))

        if (epoch + 1) % 10 == 0:
            print(
                f'Training Loss: {np.mean(train_loss_per_loader):.4f} Validation Loss: {np.mean(val_loss_per_loader):.4f}')
            print(
                f'Training Acc: {best_training_acc:.2f} Validation Acc: {best_validation_acc:.2f}')

            trainer.check_sparsity()

        plt.figure(dpi=300)
        plt.tight_layout()
        plt.plot(list(range(len(train_loss_per_epoch))), train_loss_per_epoch, label='train')
        plt.plot(list(range(len(val_loss_per_epoch))), val_loss_per_epoch, label='val')
        plt.legend(loc="upper left")
        plt.savefig('loss.png')

        plt.figure(dpi=300)
        plt.tight_layout()
        plt.plot(list(range(len(train_acc_per_epoch))), train_acc_per_epoch, label='train')
        plt.plot(list(range(len(val_acc_per_epoch))), val_acc_per_epoch, label='val')
        plt.legend(loc="upper left")
        plt.savefig('acc.png')

