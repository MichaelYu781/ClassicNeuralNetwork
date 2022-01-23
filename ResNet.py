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

    def __init__(self, in_, out_):
        super().__init__()
        self.fit_ = nn.Sequential(conv3x3(in_, out_),
                                  nn.ReLU(inplace=True),
                                  conv3x3(out_, out_)
                                  )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        fx = self.fit_(x)
        hx = x + self.relu(identity + fx)
        return hx


if __name__ == '__main__':
    conv3x3(2, 10)
    print(conv1x1(2, 10, 1)[1].weight)
