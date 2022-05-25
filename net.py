import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.prune as prune

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = models.resnet18(pretrained=True).to(device)


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


def params_to_prune(model):
    resnet_layers = [*model.model]
    prune_params = (
        (resnet_layers[0], 'weight'),
        ([*resnet_layers[4]][0].conv1, 'weight'),
        ([*resnet_layers[4]][0].conv2, 'weight'),
        ([*resnet_layers[4]][1].conv1, 'weight'),
        ([*resnet_layers[4]][1].conv2, 'weight'),
        ([*resnet_layers[5]][0].conv1, 'weight'),
        ([*resnet_layers[5]][0].conv2, 'weight'),
        ([*[*resnet_layers[5]][0].downsample][0], 'weight'),
        ([*resnet_layers[5]][1].conv1, 'weight'),
        ([*resnet_layers[5]][1].conv2, 'weight'),
        ([*resnet_layers[6]][0].conv1, 'weight'),
        ([*resnet_layers[6]][0].conv2, 'weight'),
        ([*[*resnet_layers[6]][0].downsample][0], 'weight'),
        ([*resnet_layers[6]][1].conv1, 'weight'),
        ([*resnet_layers[6]][1].conv2, 'weight'),
        ([*resnet_layers[7]][0].conv1, 'weight'),
        ([*resnet_layers[7]][0].conv2, 'weight'),
        ([*[*resnet_layers[7]][0].downsample][0], 'weight'),
        ([*resnet_layers[7]][1].conv1, 'weight'),
        ([*resnet_layers[7]][1].conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight')
    )
    return prune_params


class unfreeze_pruned_weights(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    def compute_mask(self, t, default_mask):
        mask = torch.ones_like(default_mask)
        return mask
    def apply_mask(self, module):
        orig = getattr(module, self._tensor_name + "_orig")
        mask = torch.ones_like(orig)
        pruned_tensor = mask * orig
        return pruned_tensor

def unfreeze_pruning(params_to_prune):
    for module, name in params_to_prune:
        # prune.identity(module, name)
        unfreeze_pruned_weights.apply(module, name)
        # print(module.weight_mask)

class Unfreeze_weights:
    def __init__(self, params_to_prune):
        self.paramsto_prune = params_to_prune
    def __call__(self, module, module_in, module_out):
        pass



if __name__ == '__main__':
    resnet = resnet_for_cifar()
    resnet.load_state_dict(torch.load('model.pth', map_location='cpu'))
    # for name, child in resnet.named_children():
    #     print(f'Name: {name} | Child: {child}')
    #     print("**********************")

    # layers = [*resnet.resnet.children()]
    # print(layers)
    prune.global_unstructured(
        params_to_prune(resnet),
        pruning_method=prune.L1Unstructured,
        amount=0.2
    )
    # print(*[*resnet.children()][0].children())
    # for name, child in resnet.children():
    #     print(f'Name: {name} | Child: {child}')
    #     print("diving into parameters:")
    #     for param in child.parameters():
    #         print(param.requires_grad)
    #         print("----------------------")
    #     print("****************************")
    # for param in [*[*resnet.children()][0].children()][0].parameters():
    #     print(param.requires_grad)
    # # print([*[*resnet.children()][0].children()][4])
    # for param in [*[*[*resnet.children()][0].children()][4].children()][0].conv1.parameters():
    #     print(param.requires_grad)
    #

    # TO CHECK FOR PRUNING
    # print(torch.sum([*[*resnet.children()][0].children()][0].weight == 0) / [*[*resnet.children()][0].children()][0].weight.nelement())
    # print([*[*resnet.children()][0].children()][0].weight.grad)
    # print([*[*[*resnet.children()][0].children()][4].children()][0].conv1.weight.grad)