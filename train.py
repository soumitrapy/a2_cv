from preprocessing import BSD500Dataset, load_data
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
import numpy as np
from utils import show_images

from models.simplecnn import SimpleCNN
from models.hed import HED, CBCLoss

def predict(model, dl1, criterion=None, device = 'cpu'):
    model.eval()
    val_loss = 0.0
    preds = []
    with torch.no_grad():
        for images, labels in dl1:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            else:
                outputs = outputs.cpu()
            preds.append(outputs)

    predictions = torch.cat(preds)
    return predictions, val_loss

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
        _, val_loss = predict(model, dl[1], criterion=criterion, device = device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("Training complete!")

if __name__=="__main__":
    from torchvision import transforms
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    target_transform = transforms.ToTensor()
    ds, dl = load_data(dataset_root = './BSR/BSDS500',transform=transform, target_transform=target_transform)
    model = SimpleCNN()
    optimizer = Adam(model.parameters(), lr=1e-4)
    cbc = CBCLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #Load model
    model_path = "/home/po/MTech/2ndsem/cv/a2/models/simplecnn_cuda2025-04-03 13_15_58.157753.pth"
    model.load_state_dict(torch.load(model_path,weights_only=True, map_location=device))

    preds, testloss = predict(model, dl[2],criterion=cbc, device = device)
    
    # showing random predictions
    images = []
    model.eval()
    for i in np.random.choice(len(preds), 5, replace=False):
        img, label = ds[2][i]
        pred = model(img)
        img = img.permute(1,2,0).numpy()
        pred = pred.detach().squeeze(0).numpy()
        label = label.squeeze(0).numpy()
        images.append([img, pred, label])
    show_images(images, "predictions/SimpleCNN predictions")
    

