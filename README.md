# task-at-hand
programming task with pyotrch and CIFAR-10

# About the Task
The tasks are divided are into four stages primarily:
- Build a model to train on CIFAR10
- Freeze Initial weights of the model
- Prune the lowest 10, 20 and 30%ile of the weights
- Unfreeze the pruned weights to learn their values

# About the Pipeline
I use two classes `Trainer` and `Tester` inspired from PyTorch Lightning. This helps add custom loss functions and optimizers without having to change the training pipeline. After this task I realised that it can be useful for introducing hooks and prune models.

## Task 1
The task required me to train custom Resnet18 model on CIFAR-10 dataset. 
Below is the code snippet of the custom Resnet18 class:
```python
class resnet_for_cifar(nn.Module):  
  
    def __init__(self):  
        super().__init__()  
        # self.resnet = models.resnet18(pretrained=True).to(device)  
        self.model = nn.Sequential(*(list(resnet.children())[:-1]))  
        self.fc1 = nn.Linear(in_features=512, out_features=100, bias=True)  
        self.relu = nn.ReLU(inplace=True)  
        self.fc2 = nn.Linear(in_features=100, out_features=10, bias=True)  
  
    def forward(self, x):  
        x = self.model(x)  
        x = x.squeeze(-1)  
        x = x.squeeze(-1)  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        return x
```
Cross entropy Loss is used to compare the probability distribution between the predicted values and the target, So for the first experiment the `forward` function was returning `F.softmax(x)`, for which the results are:
![](./slides/task-at-hand.003.jpeg)
## Task 2
