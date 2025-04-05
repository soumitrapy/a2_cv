from preprocessing import BSD500Dataset, load_data
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
import numpy as np
from utils import show_images, save_model, plot_losses, show_predictions
import os
import matplotlib.pyplot as plt

from models.simplecnn import SimpleCNN
from models.hed import HED, CBCLoss, HEDLoss

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
def train_model(model, dl, optimizer, criterion, num_epochs=100, device='cpu', dest = 'ckpts/'):
    model.to(device)
    model_name = dest+type(model).__name__+device.type+str(datetime.now())[:15]+'.pth'
    train_losses, val_losses = [], []
    val_loss1 = float('inf')
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
        if val_loss<val_loss1:
            save_model(model, model_name)
            val_loss1 = val_loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        plot_losses(train_losses, val_losses)
        
    print(f"Training complete! Model saved to {model_name}")
    return model_name

def save_predictions(model, dl1, criterion=None, device = 'cpu', path='predictions'):
    os.makedirs(path, exist_ok=True)
    subdir = type(model).__name__+'_'+device.type+str(datetime.now())[:15]
    path = os.path.join(path,subdir)
    os.makedirs(path, exist_ok=True)

    predictions, testloss, imagenames, labelnames = predict(model, dl1, criterion=criterion, device = device)
    pred_images = []
    if criterion:
        print(f"Loss: {testloss}")
    for i in range(predictions.shape[0]):
        img = predictions[i].squeeze(0).cpu().numpy()
        img_path = path+'/'+imagenames[i].split('/')[-1]
        pred_images.append(img_path)
        plt.imsave(img_path, img)
    return pred_images, imagenames, labelnames
    
    
    

if __name__=="__main__":
    # Data Loading
    from torchvision import transforms
    transform = transforms.ToTensor()
    target_transform = transforms.ToTensor()
    ds, dl = load_data(dataset_root = './BSR/BSDS500',transform=transform, target_transform=target_transform)


    # Model and other things initialization
    model = HED()
    optimizer = Adam(model.parameters(), lr=1e-3)
    #cbc = CBCLoss()
    hdeloss = HEDLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model training
    # model_name = train_model(model, dl, optimizer, criterion=hdeloss, num_epochs=30, device=device)

    # Load model
    model_name = "ckpts/HEDcuda2025-04-05 07:5.pth"
    model.load_state_dict(torch.load(model_name,weights_only=True, map_location=device))
    model.to(device)

    predicted_images, images, labels = save_predictions(model, dl[2], criterion=hdeloss, device = device)
    show_predictions(predicted_images, images, labels, 10, 'hed_sample_predictions.jpg')
    
    
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
    

