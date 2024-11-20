import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision


class Meso4(nn.Module):
    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.maxPool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakuReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.maxPool2 = nn.MaxPool2d(kernel_size=(4,4))
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)

        self.dropout = nn.Droupout2d(0.5)
        self.fc1 = nn.Linear(16*8*8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, input):
        x = self.conv1(input) #(8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxPool1(x) #(8, 128, 128)

        x = self.conv2(x) #(8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxPool1(x) #(8, 64, 64)

        x = self.conv3(x) #(16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxPool1(x) #(16, 32, 32)

        x = self.conv4(x) #(16, 32, 32)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxPool2(x) #(16, 8, 8)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leakyrelu(x)

        x = self.dropout(0.5)
        x = self.fc2(x)

        return x

        
