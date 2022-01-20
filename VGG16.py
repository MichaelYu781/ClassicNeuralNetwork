#
# Created By MichaelYu on 2022-01-20
#
import torch
import torch.nn as nn
from torchinfo import summary


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_ = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.MaxPool2d(2),

                                       nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.MaxPool2d(2),

                                       nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.MaxPool2d(2),

                                       nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.MaxPool2d(2),

                                       nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.MaxPool2d(2),
                                       )
        self.clf_ = nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
                                  nn.Dropout(0.5),
                                  nn.Linear(4096, 4096), nn.ReLU(inplace=True),
                                  nn.Linear(4096, 1000), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.features_(x)
        x = x.view(-1, 512 * 7 * 7)
        output = self.clf_(x)
        return output


def main():
    data = torch.ones((10, 3, 229, 229)).float()
    net = VGG16()
    summary(net, input_size=(10, 3, 229, 229), device='cpu')
    print(net(data))


if __name__ == '__main__':
    main()
