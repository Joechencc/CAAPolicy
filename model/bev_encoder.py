import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models.resnet import resnet18
from tool.config import Configuration


class BevEncoder(nn.Module):
    def __init__(self, in_channel, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        trunk = resnet18(pretrained=False, zero_init_residual=True)

        #self.conv1 = nn.Conv2d(in_channel + 1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False) if self.cfg.ttm_module else \
            nn.Conv2d(in_channel + 1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.max_pool = trunk.maxpool

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x, 2)
        return x
