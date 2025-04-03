import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import BSD500Dataset

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

data_path = "./BSR/BSDS500"
trainds = BSD500Dataset(data_path, split='train')
testds = BSD500Dataset(data_path, split='test')
valds = BSD500Dataset(data_path, split='val')
print(f'train size: {len(trainds)}, test size: {len(testds)}, val size: {len(valds)}')

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

# Class-balanced loss function
def class_balanced_loss(output, target, beta=0.99):
    target_sum = torch.sum(target)
    total_pixels = torch.numel(target)
    
    weight_pos = (1 - beta) / (1 - beta ** target_sum)
    weight_neg = (1 - beta) / (1 - beta ** (total_pixels - target_sum))
    
    weight = target * weight_pos + (1 - target) * weight_neg
    loss = nn.BCELoss(reduction='none')(output, target)
    loss = loss * weight
    return loss.mean()

# Initialize model, optimizer
model = EdgeDetectorCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train_model(model, train_loader, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, edge_maps in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = class_balanced_loss(outputs, edge_maps)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Train the model
train_model(model, train_loader, optimizer, num_epochs=100)
