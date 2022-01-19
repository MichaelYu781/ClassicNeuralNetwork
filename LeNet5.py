#
# Created By MichaelYu on 2022-01-20
#
import torch
import torch.nn as nn
from torchinfo import summary


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, X):
        y1 = torch.tanh(self.conv1(X))
        y2 = self.pool1(y1)
        y3 = torch.tanh(self.conv2(y2))
        y4 = self.pool2(y3)

        y4 = y4.view(-1, 5 * 5 * 16)

        y5 = torch.tanh(self.fc1(y4))
        output = torch.softmax(self.fc2(y5), 1)
        return output


def main():
    data = torch.ones((10, 1, 32, 32)).float()
    net = LeNet5()
    summary(net, input_size=(10, 1, 32, 32))
    # print(net(data))


if __name__ == '__main__':
    main()
