import sys, os

#Our project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.pardir))
sys.path.append(PROJECT_ROOT)

import torch
from torchvision.models.resnet import Bottleneck, ResNet


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        a = self.conv1(x)
        b = self.bn1(a)
        c = self.relu(b)
        e1 = self.maxpool(c)

        e2 = self.layer1(e1)
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        e5 = self.layer4(e4)
        return e5


def resnet50(pretrained, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = torch.load(os.path.join(PROJECT_ROOT, 'ocelot', 'models', 'bt_rn50_ep200.torch'))
        verbose = model.load_state_dict(state_dict)
        print(verbose)
    return model