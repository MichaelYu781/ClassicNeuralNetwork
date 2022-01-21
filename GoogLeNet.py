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
        self.branch1 = BasicConv2d(in_channels, ch1x1, ker_size=1)

        # 1x1 + 3x3
        self.branch2 = nn.Sequential(BasicConv2d(in_channels, ch3x3red, kernel_size=1),
                                     BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1))

        # 1x1 + 5x5
        self.branch3 = nn.Sequential(BasicConv2d(in_channels, ch5x5, kernel_size=1),
                                     BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2))
        # pool + 1x1
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     BasicConv2d(in_channels, pool_proj,kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = torch.cat([branch1, branch2, branch3, branch4], 1)
        return output


def main():
    summary(BasicConv2d(1, 10, kernel_size=3))


if __name__ == '__main__':
    main()
