#
# Created By MichaelYu on 2022-01-20
#
import torch
import torch.nn as nn
from torchinfo import summary


class NiN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(nn.Conv2d(3, 192, (5, 5), padding=(2, 2)), nn.ReLU(inplace=True),
                                    nn.Conv2d(192, 160, (1, 1)), nn.ReLU(inplace=True),
                                    nn.Conv2d(160, 96, (1, 1)), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                                    nn.Dropout(0.25)
                                    )

        self.block2 = nn.Sequential(nn.Conv2d(96, 192, (5, 5), padding=(2, 2)), nn.ReLU(inplace=True),
                                    nn.Conv2d(192, 192, (1, 1)), nn.ReLU(inplace=True),
                                    nn.Conv2d(192, 192, (1, 1)), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                                    nn.Dropout(0.25)
                                    )

        self.block3 = nn.Sequential(nn.Conv2d(192, 192, (3, 3), padding=(1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(192, 192, (1, 1)), nn.ReLU(inplace=True),
                                    nn.Conv2d(192, 10, (1, 1)), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=(7, 7), stride=(1, 1)),

                                    nn.Softmax(dim=1)
                                    )

    def forward(self, x):
        output = self.block3(self.block2(self.block1(x)))
        return output


def main():
    data = torch.ones((10, 3, 32, 32)).float()
    net = NiN()
    summary(net, input_size=(10, 3, 32, 32), device='cpu')
    print(net(data))


if __name__ == '__main__':
    main()
