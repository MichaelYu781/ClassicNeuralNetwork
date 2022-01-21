#
# Created By MichaelYu on 2022-01-21
#
import torch
import torch.nn as nn
from torchinfo import summary


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        output = self.conv(x)
        return output


class Inception(nn.Module):
    """
    in_channels: int
    ch1x1: int
    ch3x3red: int
    ch3x3: int
    ch5x5red: int
    ch5x5: int
    pool_proj: int
    """

    def __init__(self,
                 in_channels: int,
                 ch1x1: int,
                 ch3x3red: int,
                 ch3x3: int,
                 ch5x5red: int,
                 ch5x5: int,
                 pool_proj: int,
                 ):
        super().__init__()

        # 1x1
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 + 3x3
        self.branch2 = nn.Sequential(BasicConv2d(in_channels, ch3x3red, kernel_size=1),
                                     BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1))

        # 1x1 + 5x5
        self.branch3 = nn.Sequential(BasicConv2d(in_channels, ch5x5red, kernel_size=1),
                                     BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2))
        # pool + 1x1
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
                                     BasicConv2d(in_channels, pool_proj, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = torch.cat([branch1, branch2, branch3, branch4], 1)
        return output


class AuxCLF(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.feature_ = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3),
                                      BasicConv2d(in_channels, 128, kernel_size=1))
        self.clf_ = nn.Sequential(nn.Linear(4 * 4 * 128, 1024),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(0.7),
                                  nn.Linear(1024, num_classes),
                                  )

    def forward(self, x):
        x = self.feature_(x)
        x = x.view(-1, 4 * 4 * 128)
        x = self.clf_(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, blocks=None):
        super().__init__()

        if blocks is None:
            blocks = [BasicConv2d, Inception, AuxCLF]

        conv_block = blocks[0]
        inception_block = blocks[1]
        aux_clf_block = blocks[2]

        # block1
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # block2
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # block3
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # block4
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # block5
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        # Auxclf
        self.aux1 = aux_clf_block(512, num_classes)
        self.aux2 = aux_clf_block(528, num_classes)

        # clf
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # block1
        x = self.maxpool1(self.conv1(x))

        # block2
        x = self.maxpool2(self.conv3(self.conv2(x)))

        # block3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # block4
        x = self.inception4a(x)
        aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        # block5
        x = self.inception5a(x)
        x = self.inception5b(x)

        # clf
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux1, aux2


if __name__ == '__main__':
    def test_basic_conv2d():
        summary(BasicConv2d(1, 10, kernel_size=3), device='cpu')


    def test_inception():
        data = torch.ones(10, 192, 28, 28)
        net = Inception(192, 64, 96, 128, 16, 32, 32)
        print(net(data))
        summary(Inception(192, 64, 96, 128, 16, 32, 32), device='cpu')


    def test_auxclf():
        net = AuxCLF(512, 1000)
        summary(net)


    def test_googlenet():
        data = torch.ones(10, 3, 224, 224)
        net = GoogLeNet()
        summary(net, device='cpu', depth=1)
        fc2, fc1, fc0 = net(data)
        for fc in [fc2, fc1, fc0]:
            print(fc.shape)


    test_basic_conv2d()
    test_inception()
    test_auxclf()
    test_googlenet()
