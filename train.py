from preprocessing import BSD500Dataset
from models import SimpleCNN
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


"""Data Loading"""
dataset_root = './BSR/BSDS500'
transform = transforms.Compose([
transforms.ToTensor(),
])
target_transform = transforms.ToTensor()
parts = ['train', 'val', 'test']
ds = []
dl = []
for s in parts:
    ds1 = BSD500Dataset(root=dataset_root, split=s, label='BOUNDARY', transform=transform, target_transform=target_transform)
    dl1 = DataLoader(ds1, batch_size=10)
    ds.append(ds1)
    dl.append(dl1)

# ds = BSD500Dataset(root=dataset_root, split='test', label='BOUNDARY', transform=transform, target_transform=target_transform)
# dl = DataLoader(ds, batch_size=10)

import torch.nn as nn
class CBCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, y):
        b = (1-torch.sum(y))/torch.sum(torch.ones_like(y))
        loss = torch.sum(-b*torch.log(p)*y-(1-b)*torch.log(1-p)*(1-y))
        return loss


model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
cbc = CBCLoss()

# Training loop
def train_model(model, dl, optimizer, criterion, num_epochs=100, device='cpu'):
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0

        # Training phase
        for images, labels in dl[0]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in dl[1]:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training complete!")

train_model(model=model, dl=dl, optimizer=optimizer, criterion=cbc, num_epochs=2, device='cpu')

