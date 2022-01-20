#
# Created By MichaelYu on 2022-01-20
#
import torch
import torch.nn as nn
from torchinfo import summary


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, (11, 11), (4, 4))
        self.pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv2 = nn.Conv2d(96, 256, (5, 5), padding=(2, 2))
        self.pool2 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool5 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, X):
        y1 = torch.relu(self.conv1(X))
        y2 = self.pool1(y1)

        y3 = torch.relu(self.conv2(y2))
        y4 = self.pool2(y3)

        y5 = torch.relu(self.conv3(y4))
        y6 = torch.relu(self.conv4(y5))
        y7 = torch.relu(self.conv5(y6))
        y8 = self.pool5(y7)

        y8 = y8.view(-1, 6 * 6 * 256)

        y8 = torch.dropout(y8, p=0.5, train=True)
        y9 = torch.relu(self.fc1(y8))
        y10 = torch.relu(self.fc2(y9))
        output = torch.softmax(self.fc3(y10), 1)

        return output


def main():
    data = torch.ones((10, 3, 227, 227)).float()
    net = AlexNet()
    summary(net, input_size=(10, 3, 227, 227))
    # print(net(data))


if __name__ == '__main__':
    main()
