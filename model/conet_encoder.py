import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models.resnet import resnet18


class ConetEncoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)

        self.conv1 = nn.Conv3d(in_channel + 1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Convert ResNet layers to 3D
        self.layer1 = self._create_layer3d(BasicBlock3D, 64, 64, 2)
        self.layer2 = self._create_layer3d(BasicBlock3D, 64, 128, 2, stride=2)
        self.layer3 = self._create_layer3d(BasicBlock3D, 128, 256, 2, stride=2)
        self.layer4 = self._create_layer3d(BasicBlock3D, 256, 512, 2, stride=2)

    def _create_layer3d(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # torch.Size([2, 193, 160, 160, 20])
        x = F.interpolate(x, size=(256, 256, 64), mode="trilinear", align_corners=False)
        x = self.conv1(x) # torch.Size([2, 64, 128, 128, 32])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x) # torch.Size([2, 64, 64, 64, 16])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) #([2, 256, 16, 16, 4])
        # print(x.shape)
        # x = self.layer4(x) #([2, 512, 8, 8, 2])

        x = torch.flatten(x, 2) #([2, 512, 128]) -> ([2, 256, 1024])
        return x

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out