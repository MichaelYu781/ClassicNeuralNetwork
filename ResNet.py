#
# Created By MichaelYu on 2022-01-21
#

"""
要实现以下几个类
# basicconv - conv2d + BN + ReLU ( conv3x3 conv1x1 )
# Residual Unit , Bottleneck
"""

import torch
import torch.nn as nn
from torchinfo import summary
from typing import Type, Union, List, Optional


def conv3x3(in_, out_, stride_=(1, 1), initial_zero=False):
    # 残差单元，瓶颈架构最后一层需要0初始化，单独判断
    bn = nn.BatchNorm2d(out_)
    if initial_zero:
        nn.init.constant_(bn.weight, 0)

    ret = nn.Sequential(nn.Conv2d(in_, out_, kernel_size=(3, 3),
                                  padding=(1, 1), stride=stride_, bias=False),
                        bn)

    return ret


def conv1x1(in_, out_, stride_=(1, 1), initial_zero=False):
    # 残差单元，瓶颈架构最后一层需要0初始化，单独判断
    bn = nn.BatchNorm2d(out_)
    if initial_zero:
        nn.init.constant_(bn.weight, 0)

    ret = nn.Sequential(nn.Conv2d(in_, out_, kernel_size=(1, 1),
                                  padding=(0, 0), stride=stride_, bias=False),
                        bn)

    return ret


class ResidualUnit(nn.Module):
    """
    残差单元

    """

    def __init__(self, out_: int, stride1: int = 1):
        super().__init__()

        if stride1 != 1:
            in_ = int(out_ / 2)
        else:
            in_ = out_

        # 特征图尺寸是否发生变化
        self.stride1 = stride1
        self.skipconv = conv1x1(in_, out_, stride1)

        self.fit_ = nn.Sequential(conv3x3(in_, out_, stride_=stride1),
                                  nn.ReLU(inplace=True),
                                  conv3x3(out_, out_, initial_zero=True)
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
        :param middle_out: 两层卷积层后输出参数
        :param stride1: 步长默认为1，即默认为特征图尺寸不变
        :param in_: 选填参数，如果处于 conv1 后就填写该参数
        """
        super().__init__()

        out_ = 4 * middle_out

        if in_ is None:
            if stride1 != 1:
                # 此时该瓶颈结构为该layers第一个瓶颈结构
                # conv2_x , conv3_x , conv4_x , conv5_x 之间相连的部分需要将特征图减半
                in_ = 2 * middle_out
            else:
                # 此时为layers中第一个瓶颈结构之后的瓶颈结构
                in_ = 4 * middle_out

        self.fit_ = nn.Sequential(conv1x1(in_, middle_out, stride_=stride1),
                                  nn.ReLU(inplace=True),
                                  conv3x3(middle_out, middle_out),
                                  nn.ReLU(inplace=True),
                                  conv1x1(middle_out, out_,initial_zero=True)
                                  )
        self.skipconv = conv1x1(in_, out_, stride_=stride1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.fit_(x)
        indentity = self.skipconv(x)
        hx = self.relu(fx + indentity)
        return hx


if __name__ == '__main__':
    conv3x3(2, 10)
    print(conv1x1(2, 10, 1)[1].weight)

    data = torch.ones(10, 64, 56, 56)
    conv3_x_18_0 = ResidualUnit(out_=128, stride1=2)
    shape_ = conv3_x_18_0(data).shape
    print(shape_)

    conv2_x_18_0 = ResidualUnit(out_=64)
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