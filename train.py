from preprocessing import BSD500Dataset, load_data
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
import numpy as np
from utils import show_images, save_model
import os
import matplotlib.pyplot as plt

from models.simplecnn import SimpleCNN
from hed import HED, CBCLoss, HEDLoss

def predict(model, dl1, criterion=None, device = 'cpu'):
    model.eval()
    val_loss = 0.0
    preds, imagenames, labelsnames = [], [], []
    with torch.no_grad():
        for images, labels, imnames, gTruthnames in dl1:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            preds.append(outputs[-1])
            imagenames.extend(imnames)
            labelsnames.extend(gTruthnames)

    predictions = torch.cat(preds)
    return predictions, val_loss, imagenames, labelsnames

# Training loop
def train_model(model, dl, optimizer, criterion, num_epochs=100, device='cpu'):
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0

        # Training phase
        for images, labels, imnames, gTruthnames in dl[0]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        _, val_loss, _, _ = predict(model, dl[1], criterion=criterion, device = device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("Training complete!")

def save_predictions(model, dl1, criterion=None, device = 'cpu', path='predictions'):
    os.makedirs(path, exist_ok=True)
    subdir = type(model).__name__+'_'+device.type+str(datetime.now())[:15]
    path = os.path.join(path,subdir)
    os.makedirs(path)

    predictions, testloss, imagenames, labelnames = predict(model, dl1, criterion=criterion, device = device)
    if criterion:
        print(f"Loss: {testloss}")
    for i in range(predictions.shape[0]):
        img = predictions[i].squeeze(0).cpu().numpy()
        img_path = path+'/'+imagenames[i].split('/')[-1]
        plt.imsave(img_path, img)
    
    

if __name__=="__main__":
    # Data Loading
    from torchvision import transforms
    transform = transforms.ToTensor()
    target_transform = transforms.ToTensor()
    ds, dl = load_data(dataset_root = './BSR/BSDS500',transform=transform, target_transform=target_transform)


    # Model and other things initialization
    model = HED()
    optimizer = Adam(model.parameters(), lr=1e-4)
    #cbc = CBCLoss()
    hdeloss = HEDLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Model training
    train_model(model, dl, optimizer, criterion=hdeloss, num_epochs=50, device=device)
    model_path = save_model(model, 'checkpoints/')

    # #Load model
    # model_path = "/home/po/MTech/2ndsem/cv/a2/models/simplecnn_cuda2025-04-03 13_15_58.157753.pth"
    # model.load_state_dict(torch.load(model_path,weights_only=True, map_location=device))

    save_predictions(model, dl[2], criterion=hdeloss, device = device)
    # preds, testloss = predict(model, dl[2],criterion=hdeloss, device = device)

    
    # # showing random predictions
    # images = []
    # model.eval()
    # for i in np.random.choice(len(preds), 5, replace=False):
    #     img, label = ds[2][i]
    #     pred = model(img)
    #     img = img.permute(1,2,0).numpy()
    #     pred = pred.detach().squeeze(0).numpy()
    #     label = label.squeeze(0).numpy()
    #     images.append([img, pred, label])
    # show_images(images, "predictions/hed1_predictions.jpg")
    

