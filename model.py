import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


mapping=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Define a CNN

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 1 * 7, 100)
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, 36)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 1 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

  
# saved trained model path:
PATH = '/home/rakumar/char_segmentation/model.pth'

# load back the saved model
net = Net()
print(net)
net.load_state_dict(torch.load(PATH))
