import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(Net, self).__init__()

        # Convolutional layers
        # Init_channels, channels, kernel_size, padding)
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # FC layers
        # Linear layer (64x4x4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)

        # Linear Layer (500 -> 10)
        self.fc2 = nn.Linear(500, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))

        # Flatten the image
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
