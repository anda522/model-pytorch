'''
resnet34实现：卷积层和全连接层共34层
'''
import torch
import torch.nn as nn
from torch.nn import functional as F

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
    
class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layer(64, 64, 3, 1, is_shortcut=False)
        self.layer2 = self._make_layer(64, 128, 4, 2)
        self.layer3 = self._make_layer(128, 256, 6, 2)
        self.layer4 = self._make_layer(256, 512, 3, 2)

        self.fc = nn.Linear(512, num_classes)

    
    def _make_layer(self, in_channels, out_channels, block_num, stride, is_shortcut=True):
        if is_shortcut:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = None
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)

net = ResNet()
input = torch.randn(1, 3, 224, 224)
output = net(input)
print(output.size())
