"""
# basicconv - conv2d + BN + ReLU ( conv3x3 conv1x1 )
# Residual Unit , Bottleneck
"""

import torch
import torch.nn as nn
from torchinfo import summary
from typing import Type, Union, List, Optional


def conv3x3(in_, out_, stride_=(1, 1), initial_zero=False):
    bn = nn.BatchNorm2d(out_)

    # in the final block of a ResidualUnit or BottleNeck,parameters should be zero initialized
    if initial_zero:
        nn.init.constant_(bn.weight, 0)

    ret = nn.Sequential(nn.Conv2d(in_, out_, kernel_size=(3, 3),
                                  padding=(1, 1), stride=stride_, bias=False),
                        bn)

    return ret


def conv1x1(in_, out_, stride_=(1, 1), initial_zero=False):
    bn = nn.BatchNorm2d(out_)

    # in the final block of a ResidualUnit or BottleNeck,parameters should be zero initialized
    if initial_zero:
        nn.init.constant_(bn.weight, 0)

    ret = nn.Sequential(nn.Conv2d(in_, out_, kernel_size=(1, 1),
                                  padding=(0, 0), stride=stride_, bias=False),
                        bn)

    return ret


class ResidualUnit(nn.Module):
    def __init__(self, middle_out: int, stride1: int = 1, in_: Optional[int] = None):
        super().__init__()

        if stride1 != 1:
            in_ = int(middle_out / 2)
        else:
            in_ = middle_out

        # this parameter is used to judge whether the size of features pictuers will be changed
        self.stride1 = stride1
        self.skipconv = conv1x1(in_, middle_out, stride1)

        self.fit_ = nn.Sequential(conv3x3(in_, middle_out, stride_=stride1),
                                  nn.ReLU(inplace=True),
                                  conv3x3(middle_out, middle_out, initial_zero=True)
                                  )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.fit_(x)

        identity = x
        if self.stride1 != 1:
            identity = self.skipconv(x)

        hx = self.relu(identity + fx)
        return hx


class BottleNeck(nn.Module):
    def __init__(self, middle_out, stride1: int = 1, in_: Optional[int] = None):
        """
        :param middle_out: the size of features maps after two conv
        :param stride1: default number is 1 meaning that the size of features maps won't be changed
        :param in_: optional parameters,only nedded if this block is after conv1
        """
        super().__init__()

        out_ = 4 * middle_out

        if in_ is None:
            if stride1 != 1:
                # the bootleneck should be the first block of this layer s
                # in the first block of conv2_x , conv3_x , conv4_x , conv5_x ,
                # the size of feature maps should be half sized
                in_ = 2 * middle_out
            else:
                # this is the num of input for the non-first block in layers
                in_ = 4 * middle_out

        self.fit_ = nn.Sequential(conv1x1(in_, middle_out, stride_=stride1),
                                  nn.ReLU(inplace=True),
                                  conv3x3(middle_out, middle_out),
                                  nn.ReLU(inplace=True),
                                  conv1x1(middle_out, out_, initial_zero=True)
                                  )
        self.skipconv = conv1x1(in_, out_, stride_=stride1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.fit_(x)
        indentity = self.skipconv(x)
        hx = self.relu(fx + indentity)
        return hx


def make_layers(block: Type[Union[ResidualUnit, BottleNeck]],
                middle_out,
                num_blocks: int,
                afterconv1: bool = False):
    """

    :param block: "Type" restricts that parameter "block" can only be class,
                  "Union" further restricts that only two classes in the bracket can be passed
    :param middle_out: the number is equal to one of middle_out in the block of certain layer s
    :param num_blocks: the number of blocks in layers
    :param afterconv1: "True" meaning that the layers is just after "conv1 layers"
    :return: nn.Sequentia(*layers) the layers
    """

    layers = list()

    if afterconv1:
        layers.append(block(middle_out, in_=64))
    else:
        layers.append(block(middle_out, stride1=2))
    for i in range(num_blocks - 1):
        layers.append(block(middle_out))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self,
                 block: Type[Union[ResidualUnit, BottleNeck]],
                 layers: List[int],
                 num_classes: int):
        """

        :param block: whether it is a ResidualUnit or it is a BottleNeck
        :param layers:
        """
        super().__init__()

        # layer1: conv + pool
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2), bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # layer2 - layer5 ResidualUnit / BottleNeck
        self.layer2_x = make_layers(block, 64, layers[0], afterconv1=True)
        self.layer3_x = make_layers(block, 128, layers[1])
        self.layer4_x = make_layers(block, 256, layers[2])
        self.layer5_x = make_layers(block, 512, layers[3])

        # global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # fully connected layer
        if block == ResidualUnit:
            self.fc = nn.Linear(512, num_classes)
        else:
            self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer5_x(self.layer4_x(self.layer3_x(self.layer2_x(x))))
        x = self.avgpool(x)  # (num_examples , fc , 1 , 1)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    conv3x3(2, 10)
    print(conv1x1(2, 10, 1)[1].weight)

    data = torch.ones(10, 64, 56, 56)
    conv3_x_18_0 = ResidualUnit(middle_out=128, stride1=2)
    shape_ = conv3_x_18_0(data).shape
    print(shape_)

    conv2_x_18_0 = ResidualUnit(middle_out=64)
    shape_ = conv2_x_18_0(data).shape
    print(shape_)

    data = torch.ones(10, 64, 56, 56)
    conv2_x_101_0 = BottleNeck(middle_out=64, in_=64)
    shape_ = conv2_x_101_0(data).shape
    print(shape_)

    data = torch.ones(10, 256, 56, 56)
    conv3_x_101_0 = BottleNeck(middle_out=128, stride1=2)
    shape_ = conv3_x_101_0(data).shape
    print(shape_)

    data = torch.ones(10, 512, 56, 56)
    conv3_x_101_1 = BottleNeck(middle_out=128)
    shape_ = conv3_x_101_1(data).shape
    print(shape_)

    # test for ResidualUnit not after conv1
    layer_34_conv4_x = make_layers(ResidualUnit, 256, 6, False)
    print(len(layer_34_conv4_x))
    # test for ResidualUnit after conv1
    conv2_x_34 = make_layers(ResidualUnit, 64, 3, True)
    datashape = (10, 64, 56, 56)
    summary(conv2_x_34, datashape, device='cpu', depth=1)

    # test for ResNet
    datashape = (10, 3, 224, 224)
    res34 = ResNet(ResidualUnit, [3, 4, 6, 3], 1000)
    res101 = ResNet(BottleNeck, [3, 4, 23, 3], 1000)
    res50 = ResNet(BottleNeck, [3, 4, 6, 3], 1000)
    print('\n', '=' * 35, "res34:", sep='\n')
    summary(res34, datashape, device='cpu')
    print('\n', '=' * 35, "res101:", sep='\n')
    summary(res101, datashape, depth=1, device='cpu')
    print('\n', '=' * 35, "res50:", sep='\n')
    summary(res50, datashape, device='cpu')
