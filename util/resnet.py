import sys, os

#Our project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.pardir))
sys.path.append(PROJECT_ROOT)

import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, ResNet


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


def resnet50(pretrained_path: str, pretrained: bool = False, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)
    
    encoder1 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool
    )
    encoder2 = model.layer1
    encoder3 = model.layer2
    encoder4 = model.layer3
    encoder5 = model.layer4
    
    return encoder1, encoder2, encoder3, encoder4, encoder5