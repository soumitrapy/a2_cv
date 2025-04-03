import torch
import torch.nn as nn

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))  # Output must be in [0,1]
        return x

# Class Balanced CrossEntropy Loss
class CBCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, y):
        b = (1-torch.sum(y))/torch.sum(torch.ones_like(y))
        loss = torch.sum(-b*torch.log(p)*y-(1-b)*torch.log(1-p)*(1-y))
        return loss
