import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11,11), stride=(4,4))
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5,5), stride=(1,1), padding='same')
        self.batch_norm2 = nn.BatchNorm2d(192)

        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3,3), stride=(1,1), padding='same')
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding='same')
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding='same')

        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, X):
        # Layer 1
        X = F.relu(self.conv1(X))
        X = self.batch_norm1(X)
        X = F.max_pool2d(X, kernel_size=(3,3), stride=(2,2))

        # Layer 2
        X = F.relu(self.conv2(X))
        X = self.batch_norm2(X)
        X = F.max_pool2d(X, kernel_size=(3,3), stride=(2,2))

        # Layer 3
        X = F.relu(self.conv3(X))

        # Layer 4
        X = F.relu(self.conv4(X))

        # Layer 5
        X = F.relu(self.conv5(X))
        X = F.max_pool2d(X, kernel_size=(3,3), stride=(3,3)) # not stride of 2 because of 4096 neurons in Layer 6

        # Layer 6
        X = X.view(-1,4096) # flatten
        X = F.relu(self.fc1(X))
        X = self.dropout(X)

        # Layer 7
        X = self.fc2(X)
        X = self.dropout(X)

        return F.log_softmax(X, dim = 1)