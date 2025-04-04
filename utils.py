import matplotlib.pyplot as plt
from datetime import datetime
import torch
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat


def show_images(images, figname="predictions/SimpleCNN_predictions.jpg"):
    n, k = len(images), len(images[0])
    fig, axes = plt.subplots(n, k, figsize=(10, 50))
    for i, imgs in enumerate(images):
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()
def show_predictions(pred_images,images, labels, k=10, figname='sample_predictions.jpg', figsize=(10,50)):
    indices = np.random.choice(np.arange(len(pred_images)), k)
    fig, axes = plt.subplots(k, 3, figsize=figsize)
    for i, j in enumerate(indices):
        img = np.array(Image.open(images[j]))
        pred = np.array(Image.open(pred_images[j]))
        if pred.shape!=img.shape:
            pred = np.array(Image.open(pred_images[j]).rotate(-90,expand=True))
        axes[i,0].imshow(img)
        axes[i,1].imshow(pred)
        raw_gt = loadmat(labels[j])
        gTruth = raw_gt['groundTruth'][0][0][0][0][1]
        gTruth = (gTruth>0).astype(float)
        axes[i,2].imshow(gTruth)
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()

    
    
    
    

def save_model(model, model_name):
	device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
	model.to('cpu')
	torch.save(model.state_dict(), model_name)
	print(f"model saved to {model_name}")
	model.to(device)
	return model_name

def plot_losses(train_losses, val_losses, figname = 'loss_plot.png'):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


