import torchvision.datasets.cifar as cifar
from torch.utils.data import DataLoader as dl
import torchvision.transforms as transforms

class dataloaders:
    def __init__(self, train_bs=128, test_bs=100):
        self.train_bs = train_bs
        self.test_bs = test_bs

    def return_train_transform(self):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    def return_traindl(self):
        trainset = cifar.CIFAR10(root='./cifar', download=True, train=True, transform=self.return_train_transform())
        trainloader = dl(trainset, batch_size=self.train_bs, shuffle=True)
        return trainloader

    def return_test_transform(self):
        return transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    def return_testdl(self):
        testset = cifar.CIFAR10(root='./cifar', download=True, train=False, transform=self.return_test_transform())
        testloader = dl(testset, batch_size=self.test_bs, shuffle=True)
        return testloader

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 3 X 32 X 32
if __name__ == '__main__':
    best_acc = 0
    start_epoch = 0


